nvcc -std=c++20 test_common_base_types.cu

nvcc -std=c++20 -arch=sm_89 --expt-relaxed-constexpr -o test_common_base_types test_common_base_types.cu -lcuda -lcudart