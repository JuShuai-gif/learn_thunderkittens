#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/*
    CUDA 专用的线程池（ThreadPool）

    设计目标：
    - 每个 GPU 对应一个常驻 CPU 线程
    - 每个线程只在初始化时调用一次 cudaSetDevice
    - 后续反复向所有 GPU 并行派发任务，避免频繁切换 device 带来的巨大开销

    使用示例：

    // 构造：传入 device id 列表
    KittensClub club(device_ids, NUM_DEVICES);

    // 向所有 GPU 线程派发任务（无需在 task 内再 set device）
    club.execute([&](int dev_idx, cudaStream_t stream) {
        int dev;
        cudaGetDevice(&dev);
        if (dev != dev_idx) {
            fprintf(stderr, "Device mismatch: expected %d, got %d\n", dev_idx, dev);
            exit(1);
        }
    });
*/
class KittensClub {
public:
    // 构造函数：使用默认 stream（null stream）
    __host__ inline KittensClub(const int *device_ids, const int num_devices);
    // 构造函数：为每个 device 显式指定一个 CUDA stream
    __host__ inline KittensClub(const int *device_ids, const cudaStream_t *streams, const int num_devices);
    // 析构函数：停止线程池并等待所有 worker 线程退出
    __host__ inline ~KittensClub();

    // 向所有 worker 线程派发一个任务，并阻塞等待全部完成
    // task 参数：
    //   - int            : worker_id / device index
    //   - cudaStream_t   : 该 device 对应的 CUDA stream
    __host__ inline void execute(std::function<void(int, cudaStream_t)> task);

private:
    /* =======================
       状态与控制变量
       ======================= */

    // 线程池是否停止（析构时置 true）
    bool stop;

    // 每个 worker 是否有可执行任务
    // task_available[i] == true 表示第 i 个线程可以取任务
    std::vector<bool> task_available;

    // 已完成当前任务的 worker 数量
    int n_task_done;

    /* =======================
       线程池核心结构
       ======================= */

    // CPU worker 线程池（一个线程对应一个 GPU）
    std::vector<std::thread> workers;
    
    // 每个 device 使用的 CUDA stream
    std::vector<cudaStream_t> streams;
    
    // worker 线程主循环入口
    // worker_id : 线程索引（也是 device index）
    // device_id : 实际 CUDA device id
    __host__ inline void worker(int worker_id, int device_id);

    // 当前派发给所有线程的任务
    // execute() 会更新该对象
    std::function<void(int, cudaStream_t)> current_task;

    /* =======================
       同步原语
       ======================= */

    // 保护共享状态的互斥锁
    std::mutex mutex;

    // 通知 worker：有新任务可执行
    std::condition_variable cond_task_available;

    // 通知 execute()：所有 worker 已完成任务
    std::condition_variable cond_task_done;
};
    
/* ============================================================
   构造函数：使用默认 CUDA stream
   ============================================================ */
__host__ inline KittensClub::KittensClub(const int *device_ids, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        // 初始状态下没有任务
        task_available.push_back(false);
        
        // 使用默认 stream（null stream）
        streams.push_back(0); // Use default stream (null stream)
        
        // 创建 worker 线程
        // 注意：每个线程绑定一个 device
        workers.emplace_back([this, dev_idx, device_ids] { worker(dev_idx, device_ids[dev_idx]); });
    }
}

/* ============================================================
   构造函数：使用用户提供的 CUDA streams
   ============================================================ */
__host__ inline KittensClub::KittensClub(const int *device_ids, const cudaStream_t *streams_in, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        task_available.push_back(false);
        // 保存用户指定的 stream
        streams.push_back(streams_in[dev_idx]);
        // 启动 worker 线程
        workers.emplace_back([this, dev_idx, device_ids] { worker(dev_idx, device_ids[dev_idx]); });
    }
}
    
/* ============================================================
   析构函数：安全关闭线程池
   ============================================================ */
__host__ inline KittensClub::~KittensClub() {
    {
        // 设置 stop 标志，通知所有 worker 退出
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    // 唤醒所有正在等待任务的 worker
    cond_task_available.notify_all();

    // 等待所有线程安全退出
    for (std::thread &worker : workers) {
        worker.join();
    }
}
    
/* ============================================================
   向所有 GPU 派发任务，并等待完成
   ============================================================ */
__host__ inline void KittensClub::execute(std::function<void(int, cudaStream_t)> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        // 设置当前任务
        current_task = task;
        // 标记所有 worker 都有任务可执行
        for (size_t i = 0; i < task_available.size(); ++i)
            task_available[i] = true;
    }
    // 唤醒所有 worker
    cond_task_available.notify_all();
    {
        // 等待所有 worker 完成任务
        std::unique_lock<std::mutex> lock(mutex);
        cond_task_done.wait(lock, [this] { return n_task_done == workers.size(); });
        // 重置完成计数器，供下一次 execute 使用
        n_task_done = 0;
    }
}

/* ============================================================
   worker 线程主循环
   ============================================================ */
__host__ inline void KittensClub::worker(int worker_id, int device_id) {
    // ⚠️ 关键优化点：
    // 每个线程只在启动时调用一次 cudaSetDevice
    // 避免在 execute 中反复切换 device（极其昂贵）
    cudaSetDevice(device_id); // done once and never again! This saves a LOT of time
    while (true) {
        std::function<void(int, cudaStream_t)> task;
        {
            std::unique_lock<std::mutex> lock(mutex);

            // 等待：
            //   1. 有新任务
            //   2. 或线程池被要求停止
            cond_task_available.wait(lock, [this, worker_id] { return stop || task_available[worker_id]; });
            
            // 如果线程池停止，直接退出线程
            if (stop)
                return;
            // 拷贝当前任务
            task = current_task;
            // 标记该 worker 已取走任务
            task_available[worker_id] = false;
        }
        // 执行任务（无锁状态）
        // worker_id 通常作为 device index 使用
        task(worker_id, streams[worker_id]);
        {
            // 更新完成计数
            // 这里加锁的开销大约 10 微秒
            std::lock_guard<std::mutex> lock(mutex); // adds about 10 microseconds overhead
            ++n_task_done;
            // 如果这是最后一个完成的 worker，通知 execute()
            if (n_task_done == workers.size())
                cond_task_done.notify_one();
        }
    }
}
