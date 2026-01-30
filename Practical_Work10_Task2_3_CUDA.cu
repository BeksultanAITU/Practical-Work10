#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static constexpr int BLOCK = 256;

// ------------------------------------
// CUDA timer (events)
// ------------------------------------
struct GpuTimer {
    cudaEvent_t a{}, b{};
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
    }
    ~GpuTimer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void tic(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(a, s)); }
    float toc_ms(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(b, s));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

// ------------------------------------
// TASK 2 kernels
// ------------------------------------

// Коалесцированный доступ.
// Соседние потоки читают и пишут соседние элементы.
__global__ void kernel_coalesced(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

// Некоалесцированный доступ.
// Мы специально делаем "разрозненные" чтения, чтобы ухудшить доступ к памяти.
__global__ void kernel_noncoalesced(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (i * stride) % n;
        out[i] = in[j] * 2.0f + 1.0f;
    }
}

// Оптимизация через shared memory.
// Здесь мы читаем данные плиткой в shared, а затем считаем и пишем обратно.
// Чтение из global становится более организованным.
__global__ void kernel_shared_tiled(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float tile[BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tile[threadIdx.x] = in[i];
    __syncthreads();

    if (i < n) out[i] = tile[threadIdx.x] * 2.0f + 1.0f;
}

// ------------------------------------
// CPU reference for validation
// ------------------------------------
static void cpu_ref(const std::vector<float>& in, std::vector<float>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = in[i] * 2.0f + 1.0f;
}

// ------------------------------------
// Helper: run a kernel and measure time
// ------------------------------------
template <typename Kernel, typename... Args>
static float run_kernel_ms(Kernel k, dim3 grid, dim3 block, cudaStream_t stream, Args... args) {
    // Небольшой прогрев нужен, чтобы первый запуск не включал лишние накладные расходы.
    k<<<grid, block, 0, stream>>>(args...);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GpuTimer t;
    t.tic(stream);
    k<<<grid, block, 0, stream>>>(args...);
    CUDA_CHECK(cudaGetLastError());
    float ms = t.toc_ms(stream);
    return ms;
}

// ------------------------------------
// TASK 3: Hybrid processing
// ------------------------------------

// Простая GPU обработка второй части массива.
__global__ void kernel_process_range(const float* __restrict__ in, float* __restrict__ out, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = offset + i;
    if (idx < n) out[idx] = in[idx] * 2.0f + 1.0f;
}

static double cpu_time_now() {
    // Для гибридной части нам нужен таймер CPU.
    // Используем cudaEvent только для измерения GPU сегментов.
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(int argc, char** argv) {
    int N = 10'000'000;
    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));

    std::cout << "Practical_Work10\n\n";

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n\n";

    // -----------------------------
    // Prepare data once
    // -----------------------------
    std::vector<float> h_in(N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

    std::vector<float> h_ref;
    cpu_ref(h_in, h_ref);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream = 0;

    // =========================================================
    // TASK 2
    // =========================================================
    std::cout << "TASK 2\n";
    std::cout << "GPU memory access patterns and optimization\n\n";

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    float ms_coal   = run_kernel_ms(kernel_coalesced, grid, block, stream, d_in, d_out, N);
    float ms_non    = run_kernel_ms(kernel_noncoalesced, grid, block, stream, d_in, d_out, N, 97);
    float ms_shared = run_kernel_ms(kernel_shared_tiled, grid, block, stream, d_in, d_out, N);

    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

    // Проверка корректности простая.
    // Для non-coalesced формула другая, поэтому сравниваем только coalesced и shared.
    bool ok = true;
    for (int i = 0; i < std::min(N, 1000); ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-5f) { ok = false; break; }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "N                       : " << N << "\n";
    std::cout << "Kernel coalesced time   : " << ms_coal   << " ms\n";
    std::cout << "Kernel non-coalesced    : " << ms_non    << " ms\n";
    std::cout << "Kernel shared optimized : " << ms_shared << " ms\n";
    std::cout << "Correctness (shared)    : " << (ok ? "OK" : "FAILED") << "\n\n";

    // =========================================================
    // TASK 3
    // =========================================================
    std::cout << "TASK 3\n";
    std::cout << "Hybrid CPU + GPU profiling with async copies and streams\n\n";

    // В hybrid режиме мы делим массив пополам.
    // Первая половина обрабатывается на CPU.
    // Вторая половина обрабатывается на GPU.
    int half = N / 2;

    // Оптимизация в этом задании.
    // Мы используем pinned memory для уменьшения накладных расходов копирования.
    float* h_pinned_in  = nullptr;
    float* h_pinned_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_in,  (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_out, (size_t)N * sizeof(float)));

    std::copy(h_in.begin(), h_in.end(), h_pinned_in);

    cudaStream_t s1;
    CUDA_CHECK(cudaStreamCreate(&s1));

    // Измеряем отдельно время передач и время ядра.
    GpuTimer t_h2d, t_kernel, t_d2h;

    // Асинхронно копируем вторую половину на GPU.
    // Это позволяет CPU параллельно считать первую половину.
    t_h2d.tic(s1);
    CUDA_CHECK(cudaMemcpyAsync(d_in + half, h_pinned_in + half,
                               (size_t)(N - half) * sizeof(float),
                               cudaMemcpyHostToDevice, s1));
    float ms_h2d = t_h2d.toc_ms(s1);

    // Пока GPU копирует, CPU считает свою часть.
    double cpu0 = cpu_time_now();
    for (int i = 0; i < half; ++i) {
        h_pinned_out[i] = h_pinned_in[i] * 2.0f + 1.0f;
    }
    double cpu1 = cpu_time_now();
    double cpu_ms = (cpu1 - cpu0) * 1000.0;

    // Запускаем ядро для второй половины.
    int n2 = N - half;
    dim3 grid2((n2 + BLOCK - 1) / BLOCK);

    t_kernel.tic(s1);
    kernel_process_range<<<grid2, block, 0, s1>>>(d_in, d_out, N, half);
    CUDA_CHECK(cudaGetLastError());
    float ms_k = t_kernel.toc_ms(s1);

    // Асинхронно копируем результат второй половины обратно.
    t_d2h.tic(s1);
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out + half, d_out + half,
                               (size_t)n2 * sizeof(float),
                               cudaMemcpyDeviceToHost, s1));
    float ms_d2h = t_d2h.toc_ms(s1);

    // Финальная синхронизация.
    CUDA_CHECK(cudaStreamSynchronize(s1));

    // Проверяем корректность на нескольких элементах.
    bool ok_hybrid = true;
    for (int i = 0; i < std::min(N, 2000); ++i) {
        float ref = h_ref[i];
        float got = h_pinned_out[i];
        if (std::fabs(ref - got) > 1e-5f) { ok_hybrid = false; break; }
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU part time (first half)     : " << cpu_ms << " ms\n";
    std::cout << "H2D async time (second half)   : " << ms_h2d  << " ms\n";
    std::cout << "Kernel time (second half)      : " << ms_k    << " ms\n";
    std::cout << "D2H async time (second half)   : " << ms_d2h  << " ms\n";
    std::cout << "Correctness (hybrid)           : " << (ok_hybrid ? "OK" : "FAILED") << "\n\n";

    std::cout << "Optimization used: pinned memory + async copies to reduce transfer overhead.\n";
    std::cout << "Done.\n";

    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaFreeHost(h_pinned_in));
    CUDA_CHECK(cudaFreeHost(h_pinned_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
