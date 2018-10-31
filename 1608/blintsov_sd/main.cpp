#include <iostream>
#include <ctime>
#include <cmath>
#include "mpi.h"

double current_function(double x) { return cos(x)*sin(x)-tan(x)*48*x*x+2*x-x; }

double find_integral(double a, double b, int steps) {
    clock_t t0;
    clock_t t1;

    t0 = clock();

    double result = .0;

    double step = (b-a)/steps;
    for (int x(0); x <= steps; ++x)
        result += current_function(a + step * (x + 0.5f));
    result *= step;

    t1 = clock();

    std::cout << "---NON PARALLEL PROGRAM---\n";
    std::cout << "result = " << result << "\n";
    std::cout << "time = " << (double)(t1 - t0) / CLOCKS_PER_SEC << "\n\n";

    return result;
}

double find_integral_parallel(double a, double b, int steps) {
    clock_t t0;
    clock_t t1;

    double step;
    double result = .0;
    double result_mpi = .0;

    // MPI definitions
    int ProcRank, ProcNum;

    // MPI initialisations
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0)
        t0 = clock();

    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if ((a <= b) && (steps > 0)) {
        step = (b - a) / double(steps);

        for (int i(ProcRank); i <= steps; i += ProcNum)
            result += current_function(a + step * (i + 0.5f));
        result *= step;

        MPI_Reduce(&result, &result_mpi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (ProcRank == 0) {
            t1 = clock();

            std::cout << "---PARALLEL PROGRAM---\n";
            std::cout << "result_mpi = " << result_mpi << "\n";
            std::cout << "time = " << (double)(t1 - t0) / CLOCKS_PER_SEC << "\n";
        }
    }

    MPI_Finalize();

    return result;
}

int main() {
    find_integral(-100, 50000000, 10000000);
    find_integral_parallel(-100, 50000000, 10000000);
    return 0;
}