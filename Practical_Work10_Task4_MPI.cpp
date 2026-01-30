#include <mpi.h>
#include <algorithm>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static void make_counts_displs(int N, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);

    int base = N / size;
    int rem  = N % size;

    for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + counts[i - 1]);
    }
}

static void local_aggregates(const std::vector<double>& a, double& sum, double& mn, double& mx) {
    sum = 0.0;
    mn = DBL_MAX;
    mx = -DBL_MAX;
    for (double x : a) {
        sum += x;
        mn = std::min(mn, x);
        mx = std::max(mx, x);
    }
}

static void run_case(const std::string& title, int N_total, int rank, int size) {
    if (rank == 0) {
        std::cout << title << "\n";
        std::cout << "N_total = " << N_total << ", processes = " << size << "\n";
    }

    std::vector<double> full;
    if (rank == 0) {
        full.resize(N_total);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < N_total; ++i) full[i] = dist(rng);
    }

    std::vector<int> counts, displs;
    make_counts_displs(N_total, size, counts, displs);

    std::vector<double> local(counts[rank]);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Scatterv распределяет массив корректно при любом N_total.
    MPI_Scatterv(
        rank == 0 ? full.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        local.data(),
        counts[rank],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double local_sum, local_min, local_max;
    local_aggregates(local, local_sum, local_min, local_max);

    // Reduce собирает итог только на root.
    double sum_r = 0.0, min_r = 0.0, max_r = 0.0;
    MPI_Reduce(&local_sum, &sum_r, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &min_r, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &max_r, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Allreduce возвращает результат всем процессам.
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    double sum_a = 0.0, min_a = 0.0, max_a = 0.0;
    MPI_Allreduce(&local_sum, &sum_a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &min_a, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &max_a, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t3 = MPI_Wtime();

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "MPI_Reduce time   : " << (t1 - t0) << " s\n";
        std::cout << "MPI_Allreduce time: " << (t3 - t2) << " s\n";
        std::cout << "Sum / Min / Max   : " << sum_r << " / " << min_r << " / " << max_r << "\n";
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N_strong = 10'000'000;
    int N_weak_per_proc = 2'000'000;

    if (argc >= 2) N_strong = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) N_weak_per_proc = std::max(1, std::atoi(argv[2]));

    if (rank == 0) {
        std::cout << "Practical_Work10\n\n";
        std::cout << "TASK 4\n";
        std::cout << "MPI scalability: sum, min, max\n\n";
    }

    // Strong scaling.
    // Здесь общий размер массива фиксирован.
    // Увеличение процессов должно уменьшать локальную работу.
    run_case("Strong scaling case", N_strong, rank, size);

    // Weak scaling.
    // Здесь объём работы на один процесс примерно одинаковый.
    // Общий размер растёт вместе с числом процессов.
    int N_weak_total = N_weak_per_proc * size;
    run_case("Weak scaling case", N_weak_total, rank, size);

    if (rank == 0) {
        std::cout << "Done.\n";
    }

    MPI_Finalize();
    return 0;
}
