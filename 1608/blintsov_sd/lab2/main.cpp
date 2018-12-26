#include <iostream>
#include <ctime>
#include <cmath>
#include "mpi.h"

const int MAX_RAND_NUMBER = 10;
bool message_showed = false;


// Indexes- количество исходящих дуг из вершины
void FillIndexes(int* arr, const int length) {
    arr[0] = length - 1;
    for (int i = 1; i < length; i++)
        arr[i] = 1;
//    for (int i = 0; i < length; i++) {
//        std::cout << "================================================\n";
//        std::cout << " INDEXES Arr[" << i << "] = " << arr[i] << std::endl;
//        std::cout << "================================================\n\n";
//    }
}


// Edges - матрица инцидентности (номера вершин для которых дуги являются входящими)
void FillEdges(int* arr, const int length) {
    const int half_length = length / 2;

    for (int i = 0; i < half_length; i++)
        arr[i] = i + 1;

    for (int i = half_length; i < length; i++)
        arr[i] = 0;

//
//    for (int i = 0; i < length; i++) {
//        std::cout << "================================================\n";
//        std::cout << " EDGES Arr[" << i << "] = " << arr[i] << std::endl;
//        std::cout << "================================================\n\n";
//    }
}


int main(int argc, char** argv) {

    int rank, proc_amount;

    MPI_Comm star_comm;
    int *graph_indexes = nullptr, *graph_edges = nullptr;

    int *vector, data = 0, sum = 0;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &proc_amount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    graph_indexes = new int[proc_amount];
    graph_edges = new int[(proc_amount -1) * 2];

    FillIndexes(graph_indexes, proc_amount);
    FillEdges(graph_edges, (proc_amount -1) * 2);

    if (rank == 0 && !message_showed) {
        message_showed = true;
        std::cout << "================================================\n";
        std::cout << "Количество вершин графа:  " << proc_amount << std::endl;
        for (int i = 0; i < proc_amount; i++)
            std::cout << "Для " << i << "-ой(ей) вершины существует = " << graph_indexes[i] << " дуга/дуги/дуг" << std::endl;
        std::cout << "Последовательный список дуг графа: " << std::endl;
        for (int i = 0; i < proc_amount * 2 - 2; i++)
            std::cout << i << "-ая дуга = " << graph_edges[i] << "-ая вершина" << std::endl;
        std::cout << "================================================\n\n";
    }


    MPI_Graph_create(MPI_COMM_WORLD, proc_amount, graph_indexes, graph_edges, 1, &star_comm);


    if (rank == 0) {
        vector = new int[proc_amount - 1];
        srand((unsigned)time(0));

        for (int i = 0; i < proc_amount - 1; i++)
            vector[i] = rand() % MAX_RAND_NUMBER;

        for (int i = 1; i < proc_amount; i++)
            MPI_Send(&vector[i - 1], 1, MPI_INT, i, 0, star_comm);
    } else {
        MPI_Recv(&data, 1, MPI_INT, 0, 0, star_comm, &status);
        std::cout << "Ранк " << rank << " получил сообщение: " << data << std::endl;
    }

    MPI_Reduce(&data, &sum, 1, MPI_INT, MPI_SUM, 0, star_comm);

    if (rank == 0)
        std::cout << "Сумма сообщений: " << sum << std::endl;

    MPI_Finalize();

    return 0;
}
