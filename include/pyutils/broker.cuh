/**
 * @file broker.cuh
 * @brief 多进程数据交换与同步工具
 *
 * 本文件提供了 KittensBroker 类，用于在多进程环境中进行
 * 高效的数据交换与执行同步。
 *
 * 该 Broker 基于 POSIX 提供的进程间通信（IPC）机制实现，
 * 包括：
 *   - 共享内存（POSIX shared memory）
 *   - 信号量（POSIX semaphores）
 *   - 套接字（sockets）
 *
 * 主要应用场景为多 GPU / 多进程运行环境，例如：
 *   - 每个进程绑定一个本地 GPU（local rank）
 *   - 不同进程之间需要交换控制信息或少量数据
 *   - 在关键阶段进行跨进程同步（barrier / handshake）
 *
 * 该设计适用于单节点（single-node）多进程模型，
 * 常见于：
 *   - 多卡推理 / 训练
 *   - 多进程 CUDA / NCCL 启动前后的协同控制
 *
 * @note
 * - 本实现依赖 POSIX IPC 机制，仅适用于类 Unix 系统（Linux 等）
 * - 所有参与通信的进程必须运行在同一物理节点上
 * - 不适用于跨节点（multi-node）分布式通信
 */
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <semaphore.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #error "KittensBroker is not supported on Windows"
#endif


namespace kittens {

namespace detail {
namespace broker {

/* 
这段代码实现了一个多进程间通信和同步的框架，主要用于：

1. 通过共享内存实现进程间数据交换
2. 使用屏障同步机制协调多个进程的执行
3. 通过Unix域套接字进行进程间通信
*/

// 最大本地进程数量
static constexpr int MAX_LOCAL_WORLD_SIZE = 72;
// 每个进程在vault中分配的数据大小（字节），用于存储CUDA IPC内存句柄
static constexpr int VAULT_SIZE_PER_RANK = 64;  // sizeof(cudaIpcMemHandle_t)

// 共享内存数据结构体
struct KittensVault
{
    static constexpr int INIT_CODE = 0x43617473; // 初始化标识码，ASCII码对应 "cats"
    int init;       // 初始化标志，用于同步各个进程的初始化状态
    int barrier;    // 屏障计数器，用于同步所有进程
    int sense;      // 屏障感应标志，表示是否所有进程都已到达屏障
    
    // 数据存储区域，每个进程分配VAULT_SIZE_PER_RANK字节
    uint8_t data[MAX_LOCAL_WORLD_SIZE * VAULT_SIZE_PER_RANK];
};

// 计算共享内存大小，按4KB页面大小对齐
static constexpr int SHM_SIZE = (sizeof(KittensVault) + 4095) / 4096 * 4096;

/**
 * @brief 初始化进程同步机制
 * @param local_rank 当前进程的本地排名（0表示主进程）
 * @param vault 指向共享内存的指针
 */
__host__ inline static void init_sync(int local_rank,volatile KittensVault* vault){
    if (local_rank == 0){
        // 主进程初始化屏障资源
        vault->barrier = 0;
        vault->sense = 0;
        __sync_synchronize();// 内存屏障，确保之前的写入对所有进程可见
        vault->init = KittensVault::INIT_CODE;// 设置初始化完成标志
    }else{
        // 从进程等待主进程完成初始化
        while (vault->init != KittensVault::INIT_CODE)
        {
            usleep(1);  // 短暂休眠避免忙等待
        }
        __sync_synchronize();   // 内存屏障
    }   
}

/**
 * @brief 进程间同步函数（屏障同步）
 * @param local_world_size 本地进程总数
 * @param vault 指向共享内存的指针
 */
__host__ inline static void sync(int local_world_size,volatile KittensVault* vault){
    // 检查是否已初始化
    if (vault->init != KittensVault::INIT_CODE)
        throw std::runtime_error("KittensBroker: KittensVault not initialized");

    // 第一阶段：进程到达屏障
    int arrived = __sync_add_and_fetch(&vault->barrier, 1);     // 原子加1
    if (arrived == local_world_size) vault->sense = 1;          // 最后一个到达的进程设置sense标志
    while (!vault->sense) usleep(1);                            // 等待所有进程到达

    // 内存屏障，确保之前的所有写入操作对所有进程可见
    __sync_synchronize();

    // 第二阶段：进程离开屏障
    arrived = __sync_add_and_fetch(&vault->barrier, -1);        // 原子减1
    if (arrived == 0) vault->sense = 0;                         // 最后一个离开的进程清除sense标志
    while (vault->sense) usleep(1);                             // 等待所有进程离开
}

/**
 * @brief 创建命名共享内存
 * @param key 共享内存的键名
 * @param size 共享内存大小
 * @return 映射到进程地址空间的指针
 */
__host__ inline void *create_shm(const char *key, size_t size) {
    int shm_fd;
    // 创建共享内存对象，O_EXCL确保如果已存在则失败
    shm_fd = shm_open(key, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);

    if (shm_fd < 0) {
        if (errno == EEXIST)
            throw std::runtime_error("KittensBroker: Named shared memory already exists");
        throw std::runtime_error("KittensBroker: Failed to create shared memory");
    }
    
    // 设置共享内存大小
    if (ftruncate(shm_fd, size) != 0) {
        shm_unlink(key);        // 清理资源
        close(shm_fd);
        throw std::runtime_error("KittensBroker: Failed to truncate shared memory");
    }
    
    // 将共享内存映射到进程地址空间
    void *addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);      // 文件描述符不再需要
    if (addr == MAP_FAILED) {
        shm_unlink(key);
        throw std::runtime_error("KittensBroker: Failed to map to shared memory");
    }

    return addr;
}


/**
 * @brief 打开已存在的命名共享内存
 * @param key 共享内存的键名
 * @param size 期望的共享内存大小
 * @return 映射到进程地址空间的指针
 */
__host__ inline void *open_shm(const char *key, size_t size) {
    int shm_fd;
    // 等待共享内存被创建
    while (true) {
        shm_fd = shm_open(key, O_RDWR | O_CLOEXEC, 0);
        if (shm_fd >= 0)
            break;
        if (errno != ENOENT)    // 只有"不存在"错误才重试
            throw std::runtime_error("KittensBroker: Failed to open shared memory");
        usleep(1);  // 短暂等待后重试
    }

    struct stat shm_st;
    // 等待共享内存达到预期大小
    do {
        if (fstat(shm_fd, &shm_st) != 0) {
            shm_unlink(key);
            close(shm_fd);
            throw std::runtime_error("KittensBroker: Failed to open shared memory stats");
        }
        usleep(1);
    } while ((size_t)shm_st.st_size < size);
    
    // 映射共享内存
    void *addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);
    if (addr == MAP_FAILED) {
        shm_unlink(key);
        throw std::runtime_error("KittensBroker: Failed to map to shared memory");
    }

    return addr;
}

/**
 * @brief 删除命名共享内存（实际删除延迟到所有引用关闭后）
 * @param key 共享内存的键名
 */
__host__ inline void unlink_shm(const char *key) {
    shm_unlink(key);
}

/**
 * @brief 取消共享内存映射
 * @param addr 映射的地址指针
 * @param size 映射的大小
 */
__host__ inline void unmap_shm(void *addr, size_t size) {
    munmap(addr, size);
}

/**
 * @brief 创建Unix域套接字
 * @param key 基础键名
 * @param local_rank 当前进程的本地排名，用于生成唯一套接字名
 * @return 套接字文件描述符
 */
__host__ inline int create_socket(const char *key, int local_rank) {
    int sock_fd;
    // 创建数据报套接字（UDP风格）
    if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0)) < 0)
        throw std::runtime_error("KittensBroker: Socket creation error");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;  // Unix域套接字

    // 为每个进程生成唯一的套接字名
    char unique_key[64];
    int n = snprintf(unique_key, sizeof(unique_key), "%s%d", key, local_rank);
    if (n < 0 || n >= (int)sizeof(unique_key)) {
        close(sock_fd);
        throw std::runtime_error("KittensBroker: Socket name too long"); 
    }
    
    // 检查套接字路径长度限制
    size_t len = strnlen(unique_key, sizeof(addr.sun_path));
    if (len > (sizeof(addr.sun_path) - 1)) {
        close(sock_fd);
        throw std::runtime_error("KittensBroker: Socket name too long");
    }
    strcpy(addr.sun_path, unique_key);
    unlink(unique_key); // 确保路径不存在
    
    // 绑定套接字到文件系统路径
    if (bind(sock_fd, (struct sockaddr *)&addr, SUN_LEN(&addr)) < 0) {
        close(sock_fd);
        throw std::runtime_error("KittensBroker: Failed to bind socket");
    }

    return sock_fd;
}

/**
 * @brief 通过Unix域套接字发送文件描述符
 * @param sock_fd 发送方套接字文件描述符
 * @param data_fd 要发送的文件描述符
 * @param dst_key 目标套接字的基础键名
 * @param dst_local_rank 目标进程的本地排名
 * @param src_local_rank 源进程的本地排名（用于标识发送方）
 */
__host__ inline void send_fd(
    int sock_fd,
    int data_fd,
    const char *dst_key,
    int dst_local_rank,
    int src_local_rank
) {
    // 联合体用于存储控制消息头
    union {
      struct cmsghdr cm;
      char* control;
    } control_un;
    
    // 计算控制消息缓冲区大小
    size_t sizeof_control = CMSG_SPACE(sizeof(int));
    control_un.control = reinterpret_cast<char *>(malloc(sizeof_control));
    if (!control_un.control) {
        close(sock_fd);
        close(data_fd);
        throw std::runtime_error("KittensBroker: Failed to allocate a control buffer");
    }
    
    // 设置消息头
    struct msghdr msg {};
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;
    
    // 设置控制消息（用于传递文件描述符）
    struct cmsghdr *cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;  // 表示传递文件描述符
    memmove(CMSG_DATA(cmptr), &data_fd, sizeof(data_fd));

    // 设置目标地址
    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    char dst_unique_key[64];
    int n = snprintf(dst_unique_key, sizeof(dst_unique_key), "%s%d", dst_key, dst_local_rank);
    if (n < 0 || n >= (int)sizeof(dst_unique_key)) { 
        free(control_un.control);
        close(sock_fd);
        close(data_fd);
        throw std::runtime_error("KittensBroker: dst path too long"); 
    }
    strcpy(addr.sun_path, dst_unique_key);
    msg.msg_name = (void *)&addr;
    msg.msg_namelen = sizeof(struct sockaddr_un);
    
    // 设置数据负载（源进程排名）
    int payload = src_local_rank;
    struct iovec iov[1];
    iov[0].iov_base = &payload;
    iov[0].iov_len  = sizeof(payload);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    
    // 发送消息（支持EINTR中断重试）
    while (true) {
        ssize_t sent = sendmsg(sock_fd, &msg, 0);
        if (sent <= 0) {
            if (errno == EINTR) continue;   // 系统调用被中断，重试
            close(sock_fd);
            close(data_fd);
            free(control_un.control);
            throw std::runtime_error("KittensBroker: Failed to send FD over socket");
        }
        break;
    }

    free(control_un.control);
}

/**
 * @brief 从Unix域套接字接收文件描述符
 * @param sock_fd 接收方套接字文件描述符
 * @param data_fd 接收到的文件描述符存储位置
 * @param src_local_rank 发送方进程的本地排名存储位置
 */
__host__ inline void recv_fd(int sock_fd, int *data_fd, int *src_local_rank) {
    union {
      struct cmsghdr cm;
      char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int));
    control_un.control = reinterpret_cast<char *>(malloc(sizeof_control));
    if (!control_un.control) {
        close(sock_fd);
        throw std::runtime_error("KittensBroker: Failed to allocate a control buffer");
    }

    struct msghdr msg {};
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;
    
    // 准备接收数据负载（源进程排名）
    int payload = -1;
    struct iovec iov[1];
    iov[0].iov_base = &payload;
    iov[0].iov_len  = sizeof(payload);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    
    // 接收消息（支持EINTR中断重试）
    while (true) {
        ssize_t received = recvmsg(sock_fd, &msg, 0);
        if (received < 0 && errno == EINTR) {
            msg.msg_controllen = sizeof_control;
            msg.msg_iovlen = 1;
            continue;
        }
        // 检查是否接收到足够的数据
        if (received < static_cast<ssize_t>(sizeof(*data_fd))) {
            free(control_un.control);
            close(sock_fd);
            throw std::runtime_error("KittensBroker: Failed to receive data over socket");
        }
        break;
    }
    // 检查控制消息是否被截断
    if (msg.msg_flags & MSG_CTRUNC) {
        free(control_un.control);
        close(sock_fd);
        throw std::runtime_error("KittensBroker: Control data truncated");
    }
    
    // 验证控制消息格式
    struct cmsghdr *cmptr = CMSG_FIRSTHDR(&msg);
    if (!cmptr ||
        cmptr->cmsg_len != CMSG_LEN(sizeof(int)) ||
        cmptr->cmsg_level != SOL_SOCKET ||
        cmptr->cmsg_type != SCM_RIGHTS) {
        free(control_un.control);
        close(sock_fd);
        throw std::runtime_error("KittensBroker: Failed to receive data over socket");
    }
    // 提取文件描述符和源进程排名
    memmove(data_fd, CMSG_DATA(cmptr), sizeof(*data_fd));
    free(control_un.control);
    *src_local_rank = payload;
}

/**
 * @brief 删除Unix域套接字文件
 * @param key 套接字基础键名
 * @param local_rank 本地进程排名
 */
__host__ inline void unlink_socket(const char *key, int local_rank) {
    char unique_key[64];
    int n = snprintf(unique_key, sizeof(unique_key), "%s%d", key, local_rank);
    if (n < 0 || n >= (int)sizeof(unique_key))
        throw std::runtime_error("KittensBroker: Socket name too long");
    unlink(unique_key); // 删除套接字文件
}

/**
 * @brief 关闭套接字文件描述符
 * @param sock_fd 套接字文件描述符
 */
__host__ inline void close_socket(int sock_fd) {
    close(sock_fd);
}

} // namespace broker
} // namespace detail

/**
 * @brief KittensBroker工具类，用于多进程数据交换
 * 
 * 注意：该代码依赖于POSIX套接字/共享内存/信号量进行进程间通信和同步。
 * 
 * 用户主要使用的函数：
 * 
 *     KittensBroker broker(local_rank, local_world_size);
 *     broker.exchange_data(dst, src, size); // 在所有进程间交换数据
 *     broker.exchange_fds(dst, src_fd); // 在所有进程间交换文件描述符
 *     broker.broadcast_fd(dst, src_fd, src_rank); // 从源进程广播文件描述符到所有进程
 *     broker.sync(); // 等待所有进程到达此点
 */
struct KittensBroker {
    // TODO: 为每个进程组生成唯一标识
    static inline constexpr const char *SHM_KEY_ = "/kittens_broker_shm";       // 共享内存键
    static inline constexpr const char *SOCK_KEY_ = "/tmp/kittens_broker.sock"; // 套接字键

    int local_rank_;        // 本地进程排名
    int local_world_size_;  // 本地进程总数

    void *shm_raw_;         // 原始共享内存指针
    volatile detail::broker::KittensVault *shm_;    // 类型化的共享内存指针
    int sock_;              // 套接字文件描述符
    
    /**
     * @brief 构造函数，初始化多进程通信环境
     * @param local_rank 本地进程排名（0表示主进程）
     * @param local_world_size 本地进程总数
     */
    __host__ inline KittensBroker(int local_rank, int local_world_size)
        : local_rank_(local_rank), 
          local_world_size_(local_world_size),
          shm_raw_(nullptr),
          shm_(nullptr),
          sock_(-1) {
        
        // 参数验证
        if (local_rank_ < 0)
            throw std::runtime_error("KittensBroker: Local rank must be non-negative");
        if (local_rank_ >= local_world_size_)
            throw std::runtime_error("KittensBroker: Local rank is greater than local world size");
        if (local_world_size_ > detail::broker::MAX_LOCAL_WORLD_SIZE)
            throw std::runtime_error("KittensBroker: Local world size is greater than MAX_LOCAL_WORLD_SIZE");
        
        // 设置共享内存
        if (local_rank_ == 0) {
            // 主进程创建共享内存
            shm_raw_ = detail::broker::create_shm(SHM_KEY_, sizeof(detail::broker::KittensVault));
            shm_ = reinterpret_cast<volatile detail::broker::KittensVault *>(shm_raw_);
            memset(shm_raw_, 0, sizeof(detail::broker::KittensVault));  // 清零初始化
        } else {
            // 从进程打开已存在的共享内存
            shm_raw_ = detail::broker::open_shm(SHM_KEY_, sizeof(detail::broker::KittensVault));
            shm_ = reinterpret_cast<volatile detail::broker::KittensVault *>(shm_raw_);
        }
        // 初始化同步机制
        detail::broker::init_sync(local_rank_, shm_);
        detail::broker::sync(local_world_size_, shm_);
        
        // 主进程取消共享内存链接（实际删除延迟到所有引用关闭后）
        if (local_rank_ ==0)
            detail::broker::unlink_shm(SHM_KEY_);
        detail::broker::sync(local_world_size_, shm_);
        
        // 创建套接字
        sock_ = detail::broker::create_socket(SOCK_KEY_, local_rank_);
        detail::broker::sync(local_world_size_, shm_);
    }

    // 禁止拷贝构造函数和赋值运算符
    KittensBroker(const KittensBroker&) = delete;
    KittensBroker& operator=(const KittensBroker&) = delete;
    /**
     * @brief 移动构造函数
     */
    __host__ inline KittensBroker(KittensBroker&& other) noexcept
        : local_rank_(other.local_rank_),
          local_world_size_(other.local_world_size_),
          shm_raw_(other.shm_raw_),
          shm_(other.shm_),
          sock_(other.sock_) {
        other.local_rank_ = -1;
        other.local_world_size_ = -1;
        other.shm_raw_ = nullptr;
        other.shm_ = nullptr;
        other.sock_ = -1;
    }

    /**
     * @brief 销毁资源
     */
    __host__ inline void destroy() {
        if (shm_raw_) {
            detail::broker::unmap_shm(shm_raw_, sizeof(detail::broker::KittensVault));
            shm_raw_ = nullptr;
            shm_ = nullptr;
        }
        if (sock_ >= 0) {
            detail::broker::unlink_socket(SOCK_KEY_, local_rank_);
            detail::broker::close_socket(sock_);
            sock_ = -1;
        }
        local_rank_ = -1;
        local_world_size_ = -1;
    }

    /**
     * @brief 移动赋值运算符
     */
    __host__ inline KittensBroker& operator=(KittensBroker&& other) noexcept {
        if (this != &other) {
            destroy();
            local_rank_ = other.local_rank_;
            local_world_size_ = other.local_world_size_;
            shm_raw_ = other.shm_raw_;
            shm_ = other.shm_;
            sock_ = other.sock_;
            other.local_rank_ = -1;
            other.local_world_size_ = -1;
            other.shm_raw_ = nullptr;
            other.shm_ = nullptr;
            other.sock_ = -1;
        }
        return *this;
    }

    /**
     * @brief 析构函数
     */
    __host__ inline ~KittensBroker() {
        destroy();
    }

    /**
     * @brief 进程同步（屏障）
     * @param num_ranks 需要同步的进程数，默认为所有进程
     */
    __host__ inline void sync(int num_ranks = -1) {
        if (num_ranks == -1)
            num_ranks = local_world_size_;
        else if (num_ranks < 0 || num_ranks > local_world_size_)
            throw std::runtime_error("KittensBroker: Invalid number of ranks");

        detail::broker::sync(num_ranks, shm_);
    }

    /**
     * @brief 在所有进程间交换数据
     * @param dst_ 目标缓冲区（接收所有进程的数据）
     * @param src_ 源数据缓冲区（当前进程要发送的数据）
     * @param size 数据大小（每个进程）
     */
    __host__ inline void exchange_data(void *dst_, const void *src_, size_t size) {
        if (size > detail::broker::VAULT_SIZE_PER_RANK)
            throw std::runtime_error("KittensBroker: Size is greater than VAULT_SIZE_PER_RANK");

        uint8_t *dst = reinterpret_cast<uint8_t *>(dst_);
        const uint8_t *src = reinterpret_cast<const uint8_t *>(src_);

        // 交换数据
        sync(); // 确保所有进程一起进入
        // 将源数据复制到共享内存中自己的位置
        memcpy(const_cast<uint8_t *>(shm_->data) + local_rank_ * detail::broker::VAULT_SIZE_PER_RANK, src, size);
        sync(); // 确保所有进程一起退出
    
        // 从共享内存中收集所有进程的数据到目标缓冲区
        for (int i = 0; i < local_world_size_; i++)
            memcpy(dst + i * size, const_cast<uint8_t *>(shm_->data) + i * detail::broker::VAULT_SIZE_PER_RANK, size);
    }

    /**
     * @brief 在所有进程间交换文件描述符
     * @param dst 目标数组，接收所有进程的文件描述符
     * @param data_fd 当前进程要发送的文件描述符
     */
    __host__ inline void exchange_fds(int *dst, const int data_fd) {
        if (dst == nullptr)
            throw std::runtime_error("KittensBroker: dst is null");
        if (data_fd < 0)
            throw std::runtime_error("KittensBroker: source fd is negative");

        // 初始化目标缓冲区
        for (int i = 0; i < local_world_size_; ++i)
            dst[i] = -1;

        // 确保所有进程一起进入
        sync();

        if (local_rank_ == 0) {
            // 排名0的进程收集所有其他进程的文件描述符并分发给它们
            dst[0] = data_fd;
            // 接收其他进程的文件描述符
            for (int i = 0; i < local_world_size_ - 1; i++) {
                int received_fd;
                int src_local_rank;
                detail::broker::recv_fd(sock_, &received_fd, &src_local_rank);
                if (received_fd < 0)
                    throw std::runtime_error("KittensBroker: Failed to receive FD over socket");
                if (src_local_rank == local_rank_)
                    throw std::runtime_error("KittensBroker: Invalid source rank");
                dst[src_local_rank] = received_fd;
            }
            // 将所有文件描述符发送给其他进程
            for (int dst_local_rank = 1; dst_local_rank < local_world_size_; dst_local_rank++) {
                for (int src_local_rank = 0; src_local_rank < local_world_size_; src_local_rank++) {
                    if (dst_local_rank == src_local_rank)
                        continue;
                    detail::broker::send_fd(sock_, dst[src_local_rank], SOCK_KEY_, dst_local_rank, src_local_rank);
                }
            }
            close(dst[0]); // 不再需要自己的文件描述符
            dst[0] = -1;
        } else {
            // 其他进程发送自己的文件描述符给排名0，然后从排名0接收所有文件描述符
            detail::broker::send_fd(sock_, data_fd, SOCK_KEY_, 0, local_rank_);
            close(data_fd); // 不再需要自己的文件描述符
            for (int i = 0; i < local_world_size_ - 1; i++) {
                int received_fd;
                int src_local_rank;
                detail::broker::recv_fd(sock_, &received_fd, &src_local_rank);
                if (received_fd < 0)
                    throw std::runtime_error("KittensBroker: Failed to receive FD over socket");
                if (src_local_rank == local_rank_)
                    throw std::runtime_error("KittensBroker: Invalid source rank");
                dst[src_local_rank] = received_fd;
            }
        }

        // 确保所有进程一起退出
        sync();
    }

    __host__ inline void broadcast_fd(int *dst, const int data_fd, const int src_local_rank) {
        if (src_local_rank < 0 || src_local_rank >= local_world_size_)
            throw std::runtime_error("KittensBroker: Invalid source rank");

        // 确保所有进程一起进入
        sync();

        if (local_rank_ == src_local_rank) {
            // 源进程发送文件描述符给所有其他进程
            if (data_fd < 0)
                throw std::runtime_error("KittensBroker: Source rank has invalid FD");
            for (int dst_local_rank = 0; dst_local_rank < local_world_size_; dst_local_rank++) {
                if (dst_local_rank == src_local_rank)
                    continue;
                detail::broker::send_fd(sock_, data_fd, SOCK_KEY_, dst_local_rank, src_local_rank);
            }
            close(data_fd); // 不再需要源文件描述符
        } else {
            // 接收进程从源进程接收文件描述符
            if (!dst)
                throw std::runtime_error("KittensBroker: Destination rank has invalid buffer");
            int _src_local_rank;
            detail::broker::recv_fd(sock_, dst, &_src_local_rank);
            if (*dst < 0)
                throw std::runtime_error("KittensBroker: Failed to receive valid FD over socket");
            if (_src_local_rank != src_local_rank)
                throw std::runtime_error("KittensBroker: Invalid source rank");
        }

        // 确保所有进程一起退出
        sync();
    }
};

} // namespace kittens














