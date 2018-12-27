#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace cv;

unsigned char clamp(double val) {
    if (val < 0)
        val = 0;
    else if(val > (double)UCHAR_MAX)
        val = (double)UCHAR_MAX;
    return (unsigned char)val;
}

int main(int argc, char* argv[]) {
//    std::cout<<argc<<std::endl;
//    for (int i = 0; i<argc; i++) {
//        std::cout<<argv[i]<<std::endl;
//    }
    Mat *image = nullptr;
    Mat *lin_image = nullptr;
    Mat *res_image = nullptr;
    int img_width = 0;
    int img_height = 0;
    int img_size = 0;

    int kernel_size = 3;
    int center = kernel_size / 2;
    double sigma = 0.5f;

    unsigned char *buf  = nullptr;
    int size = 0;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int proc_num = 0;
    int proc_rank = 0;
    int height_tiles = 0;
    int width_tiles = 0;
    double *data = nullptr;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    if (proc_rank == 0) {
        image = new Mat();
        std::cout << argv[1];
        *image = imread(argv[1], IMREAD_GRAYSCALE);
        if (!image->data) {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        img_width = image->cols;
        img_height = image->rows;
        img_size = img_height * img_width;
        lin_image = new Mat(img_height, img_width, CV_8UC1);
        res_image = new Mat(img_height, img_width, CV_8UC1);

        // считаем кернел ядро фильтра гаусса
        data = new double[kernel_size * kernel_size];
        for (int i = -center; i <= center; i++) {
            for (int j = -center; j <= center; j++) {
                data[(i + center) + kernel_size * (j + center)] = pow(sqrt(2 * M_PI) * sigma, -1) * exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(sigma, 2)));
            }
        }
        start_time = MPI_Wtime();
        // делаем свёртку с этим ядром
        for (int i = 0; i < img_width; i++) {
            for (int j = 0; j < img_height; j++) {
                lin_image->data[i + j * img_width] = 0;
                for (int k_i = -center; k_i <= center; k_i++) {
                    for (int k_j = -center; k_j <= center; k_j++) {
                        int new_i = i - k_i;
                        new_i = new_i > 0 ? new_i : 0;
                        new_i = new_i < img_width ? new_i : (img_width - 1);
                        int new_j = j - k_j;
                        new_j = new_j > 0 ? new_j : 0;
                        new_j = new_j < img_height ? new_j : (img_height - 1);
                        double add = image->data[new_i + new_j * img_width] * data[(k_i + center) + (k_j + center) * kernel_size];
                        lin_image->data[i + j * img_width] = clamp(lin_image->data[i + j * img_width] + add);
                    }
                }
            }
        }
        end_time = MPI_Wtime();
        std::cout << "Line time: " << end_time - start_time << std::endl;
        // считаем оптимальные блоки на которые нужно разбить изображение
        std::vector<int> mults;
        int tmp = proc_num;
        int mul = 1;
        for (int i = 2; i < tmp;) {
            if ((tmp / mul) % i == 0) {
                mults.push_back(i);
                mul *= i;
            }
            else {
                i++;
            }
        }

        if (!mults.empty()) {
            int sum = 0;
            for (auto &el : mults) {
                sum += el;
            }
            float sumDiv2 = (float)sum / 2;
            int sum1 = 0;
            int i = 0;
            for (; i < mults.size(); i++) {
                if (sum1 + mults[i] < sumDiv2) {
                    sum1 += mults[i];
                }
                else {
                    i--;
                    break;
                }
            }
            if (fabs((float)sum1 - sumDiv2) >= fabs((float)sum1 + (float)mults[i + 1] - sumDiv2)) {
                sum1 += mults[++i];
            }
            height_tiles = 1;
            width_tiles = 1;
            if (mults.size() != 2) {
                for (int j = 0; j <= i; j++)
                    height_tiles *= mults[j];
                for (int j = i + 1; j < mults.size(); j++)
                    width_tiles *= mults[j];
            }
            else {
                height_tiles = mults[0];
                width_tiles = mults[1];
            }
        }
        else {
            height_tiles = proc_num;
            width_tiles = 1;
        }
        start_time = MPI_Wtime();
        // вычисляем какого размера блоки мы должны отослать и отсылаем их, также отсылаем размеры этих блоков и ядро фильтра
        for (int i = 0; i < width_tiles; i++) {
            for (int j = 0; j < height_tiles; j++) {
                if (i + j * width_tiles == 0)
                    continue;
                else {
                    int tile_width_begin = i * img_width / width_tiles;
                    int tile_width_end = (i + 1) * img_width / width_tiles;
                    int tile_height_begin = j * img_height / height_tiles;
                    int tile_height_end = (j + 1) * img_height / height_tiles;
                    int width_size = tile_width_end - tile_width_begin + 2 * center;
                    int height_size = tile_height_end - tile_height_begin + 2 * center;
                    int size = width_size * height_size;
                    MPI_Send(&width_size, 1, MPI_INT, i + j * width_tiles, 1, MPI_COMM_WORLD);
                    MPI_Send(&height_size, 1, MPI_INT, i + j * width_tiles, 2, MPI_COMM_WORLD);
                    buf = new unsigned char[size];
                    for(int w = 0; w < width_size; w++)
                        for (int h = 0; h < height_size; h++) {
                            int w_idx = tile_width_begin + w - center;
                            w_idx = w_idx > 0 ? (w_idx < img_width ? w_idx : (img_width - 1)) : 0;
                            int h_idx = tile_height_begin + h - center;
                            h_idx = h_idx > 0 ? (h_idx < img_height ? h_idx : (img_height - 1)) : 0;
                            buf[w + h * width_size] = image->data[w_idx + h_idx * img_width];
                        }
                    MPI_Send(buf, size, MPI_CHAR, i + j * width_tiles, 0, MPI_COMM_WORLD);
                    MPI_Send(data, kernel_size * kernel_size, MPI_DOUBLE, i + j * width_tiles, 3, MPI_COMM_WORLD);
                    delete buf;
                }
            }
        }
    }

    if (proc_rank > 0){
        MPI_Status stat;
        int width_size;
        int height_size;
        // принимаем всё что отослали и применяем ядро к блоку
        MPI_Recv(&width_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &stat);
        MPI_Recv(&height_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &stat);

        MPI_Probe(0, 0, MPI_COMM_WORLD, &stat);
        buf = new unsigned char[width_size * height_size];
        MPI_Recv(buf, width_size * height_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &stat);
        double *data = new double[kernel_size * kernel_size];
        unsigned char *res_buf = new unsigned char[width_size * height_size];
        MPI_Recv(data, kernel_size * kernel_size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &stat);
        for (int i = 0; i < width_size; i++) {
            for (int j = 0; j < height_size; j++) {
                res_buf[i + j * width_size] = 0;
                for (int k_i = -center; k_i <= center; k_i++) {
                    for (int k_j = -center; k_j <= center; k_j++) {
                        int new_i = i - k_i;
                        new_i = new_i > 0 ? new_i : 0;
                        new_i = new_i < width_size ? new_i : (width_size - 1);
                        int new_j = j - k_j;
                        new_j = new_j > 0 ? new_j : 0;
                        new_j = new_j < height_size ? new_j : (height_size - 1);
                        double add = buf[new_i + new_j * width_size] * data[(k_i + center) + (k_j + center) * kernel_size];
                        res_buf[i + j * width_size] = clamp(res_buf[i + j * width_size] + add);
                    }
                }
            }
        }
        // отсылаем обработанный блок обратно
        MPI_Send(res_buf, width_size * height_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else {
        //обрабатываем блок нулевого процесса
        int tile_width_begin = 0;
        int tile_width_end = img_width / width_tiles;
        int tile_height_begin = 0;
        int tile_height_end = img_height / height_tiles;
        for (int i = 0; i < tile_width_end; i++) {
            for (int j = 0; j < tile_height_end; j++) {
                res_image->data[i + j * img_width] = 0;
                for (int k_i = -center; k_i <= center; k_i++) {
                    for (int k_j = -center; k_j <= center; k_j++) {
                        int new_i = i - k_i;
                        new_i = new_i > 0 ? new_i : 0;
                        new_i = new_i < img_width ? new_i : (img_width - 1);
                        int new_j = j - k_j;
                        new_j = new_j > 0 ? new_j : 0;
                        new_j = new_j < img_height ? new_j : (img_height - 1);
                        double add = image->data[new_i + new_j * img_width] * data[(k_i + center) + (k_j + center) * kernel_size];
                        res_image->data[i + j * img_width] = clamp(res_image->data[i + j * img_width] + add);
                    }
                }
            }
        }
    }

    if (proc_rank == 0) {
        for (int i = 1; i < proc_num; i++) {
            //принимаем обработанные блоки, пересчитываем их размер(мы его уже забыли к этому времени) и засовываем блоки в результирующую картинку
            MPI_Status stat;
            MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
            int width_idx = stat.MPI_SOURCE % width_tiles;
            int height_idx = stat.MPI_SOURCE / width_tiles;
            int tile_width_begin = width_idx * img_width / width_tiles;
            int tile_width_end = (width_idx + 1) * img_width / width_tiles;
            int tile_height_begin = height_idx * img_height / height_tiles;
            int tile_height_end = (height_idx + 1) * img_height / height_tiles;
            int width_size = tile_width_end - tile_width_begin + 2 * center;
            int height_size = tile_height_end - tile_height_begin + 2 * center;
            unsigned char *res_buf = new unsigned char[width_size * height_size];
            MPI_Recv(res_buf, width_size * height_size, MPI_CHAR, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
            for (int w = tile_width_begin; w < tile_width_end; w++)
                for (int h = tile_height_begin; h < tile_height_end; h++)
                    res_image->data[w + h * img_width] = res_buf[(w - tile_width_begin + center) + (h - tile_height_begin + center) * width_size];
            delete[] res_buf;
        }
        end_time = MPI_Wtime();
        std::cout << "Parallel time: " << end_time - start_time << std::endl;
        // проверяем сходится ли результат линейной и параллельной версии
        bool ok = true;
        for (int i = 0; i < img_size; i++) {
            if (res_image->data[i] != lin_image->data[i]) {
                ok = false;
            }
        }
        if (ok) {
            std::cout << "TEST PASSED!" << std::endl;
        }
        else {
            std::cout << "TEST FAILED!" << std::endl;
        }
        //выводим картинки
        namedWindow("RAW IMAGE", CV_WINDOW_KEEPRATIO);
        imshow("RAW IMAGE", *image);
        namedWindow("LINE CALC", CV_WINDOW_KEEPRATIO);
        imshow("LINE CALC", *lin_image);
        namedWindow("PARALLEL CALC", CV_WINDOW_KEEPRATIO);
        imshow("PARALLEL CALC", *res_image);

        waitKey(0);
    }
    MPI_Finalize();
    return 0;
}
