#include <iostream>
#include "mpi.h"

double current_function(double x)
{
    return 2*x-x;
}

double find_integral(double a, double b, int steps) {
    double result = .0;

    double step = (b-a)/steps;
    for (double x = a; x <= b; x+=step)
        result += current_function(x-step/2);
    result *= step;
    return result;
}

int main() {
    std::cout << find_integral(1, 10, 20);
}