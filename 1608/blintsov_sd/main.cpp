#include <iostream>
#include "mpi.h"
#include <cmath>

double f(double a) {
  return (4.0 / (1.0 + a*a));
}

void find_integral(double a, double b, int steps) {
    double t1 = 0, t2 = 0, step, result = 0;

    t1 = MPI_Wtime();

    if (steps > 0) {
        step = (b-a) / (double) steps;

        for (int x(0); x <= steps; ++x)
            result += f(a + step * (x + 0.5f));
        result *= step;

        t2 = MPI_Wtime();
        std::cout << "---NON PARALLEL PROGRAM---\n";
        std::cout << result << std::endl;
        std::cout << "Time = " << t2-t1 << std::endl;
    }
}

int main(int argc, char *argv[]) {
  int ProcRank, ProcNum, i, steps = 0;
  double t1 = 0, t2 = 0, sum, a = 0, b = 0, step, step_result, result;
  bool done = false;

  std::cout << "Enter the number of intervals: ";
  std::cin >> steps;
  std::cout << "Enter a,b: ";
  std::cin >> a >> b;

// RUN NON-PARALLEL VERSION
  find_integral(a,b,steps);

// START PARALLEL VERSION
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

  while (!done) {
    if (ProcRank == 0)
      t1 = MPI_Wtime();

    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (steps > 0) {
      step = (b-a) / (double) steps;
      sum = 0.0;
      for (i = ProcRank + 1; i <= steps; i += ProcNum) {
          sum += f(a + step * (static_cast<double>(i) - 0.5));
      }
      step_result = step * sum;
      MPI_Reduce(&step_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0,
                 MPI_COMM_WORLD);
      if (ProcRank == 0) {
          t2 = MPI_Wtime();
          std::cout << "---PARALLEL PROGRAM---\n";
          std::cout << result << std::endl;
          std::cout << "Time = " << t2-t1 << std::endl;
          done = true;
      }
    }
    else done = true;
  }

  MPI_Finalize();
// END PARALLEL VERSION
  return 0;
}
