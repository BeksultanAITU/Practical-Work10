#include <omp.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static double cpu_sum_seq(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x;
    return s;
}

static double cpu_sumsq_seq(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x * x;
    return s;
}

static void cpu_stats_openmp(const std::vector<double>& a, int threads,
                             double& sum, double& mean, double& var, double& time_s)
{
    // Здесь мы измеряем только саму обработку массива.
    // Генерация данных не входит в измерение, чтобы не смешивать разные этапы.

    omp_set_num_threads(threads);

    double t0 = omp_get_wtime();

    // Редукция суммы и суммы квадратов.
    // OpenMP сам объединяет частичные результаты от разных потоков.
    double s  = 0.0;
    double s2 = 0.0;

    #pragma omp parallel for reduction(+:s,s2)
    for (int i = 0; i < (int)a.size(); ++i) {
        double x = a[i];
        s  += x;
        s2 += x * x;
    }

    double t1 = omp_get_wtime();

    sum  = s;
    mean = s / (double)a.size();

    // Дисперсия через формулу E[x^2] - (E[x])^2.
    // При небольших погрешностях может получиться отрицательное значение очень близкое к нулю.
    double ex2 = s2 / (double)a.size();
    double ex  = mean;
    var = ex2 - ex * ex;
    if (var < 0.0) var = 0.0;

    time_s = (t1 - t0);
}

static double estimate_parallel_fraction_amdahl(double speedup, int threads) {
    // Закон Амдала: S = 1 / ( (1 - p) + p / T )
    // Отсюда p = (1 - 1/S) / (1 - 1/T)
    if (threads <= 1) return 0.0;
    if (speedup <= 0.0) return 0.0;

    double S = speedup;
    double T = (double)threads;

    double num = 1.0 - 1.0 / S;
    double den = 1.0 - 1.0 / T;
    if (std::fabs(den) < 1e-12) return 0.0;

    double p = num / den;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

int main(int argc, char** argv) {
    int N = 10'000'000;
    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));

    std::cout << "Practical_Work10\n\n";

    std::cout << "TASK 1\n";
    std::cout << "OpenMP CPU stats: sum, mean, variance\n";
    std::cout << "Array size N = " << N << "\n\n";

    // Подготавливаем данные один раз.
    // Используем фиксированный seed, чтобы результаты воспроизводились.
    std::vector<double> a(N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; ++i) a[i] = dist(rng);

    // Последовательная база.
    // Это нужно для сравнения и для оценки ускорения.
    double t0 = omp_get_wtime();
    double sum_seq  = cpu_sum_seq(a);
    double sumsq_seq = cpu_sumsq_seq(a);
    double t1 = omp_get_wtime();

    double mean_seq = sum_seq / (double)N;
    double var_seq  = sumsq_seq / (double)N - mean_seq * mean_seq;
    if (var_seq < 0.0) var_seq = 0.0;

    double time_seq = (t1 - t0);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Sequential time: " << time_seq << " s\n";
    std::cout << "Sequential mean: " << mean_seq << "\n";
    std::cout << "Sequential var : " << var_seq  << "\n\n";

    // Набор потоков для эксперимента.
    // Для разных машин можно изменить.
    std::vector<int> thread_list = {1, 2, 4, 8, 16};

    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(14) << "Time(s)"
              << std::setw(14) << "Speedup"
              << std::setw(14) << "p(Amdahl)"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (int th : thread_list) {
        if (th > omp_get_max_threads()) continue;

        double sum = 0.0, mean = 0.0, var = 0.0, time_par = 0.0;
        cpu_stats_openmp(a, th, sum, mean, var, time_par);

        double speedup = time_seq / time_par;
        double p = estimate_parallel_fraction_amdahl(speedup, th);

        std::cout << std::left
                  << std::setw(10) << th
                  << std::setw(14) << time_par
                  << std::setw(14) << speedup
                  << std::setw(14) << p
                  << "\n";
    }

    std::cout << "\n";
    std::cout << "Note: p(Amdahl) is an estimated parallel fraction from measured speedup.\n";
    std::cout << "Done.\n";
    return 0;
}
