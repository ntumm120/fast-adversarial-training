#include "parallel_multiply.h"
// #include "timer.h"

#include <cstdio>
#include <cstdlib>

#define TRIALS 10
#define szRAND 1000

void tm_test(int n, int sz, int nThreads) {
    torch::Tensor a[n];

    for (int i = 0; i < n; ++i){
        a[i] = torch::randn({sz, sz});
    }

    test_matrix_multiplication();

    // torch::set_num_threads(1);

    const double sequential_start = get_wtime();
    torch::Tensor sequential_tensor = tm_sequential(a, n);
    const double sequential_end = get_wtime();

    // torch::set_num_threads(nThreads);

    const double simple_start = get_wtime();
    torch::Tensor simple_tensor = tm_simple(a, n);
    const double simple_end = get_wtime();

    // torch::set_num_threads(nThreads);

    const double complex_start = get_wtime();
    torch::Tensor complex_tensor = tm(a, n);
    const double complex_end = get_wtime();

    const double elapsed_sequential = sequential_end - sequential_start;
    const double elapsed_simple = simple_end - simple_start;
    const double elapsed_complex = complex_end - complex_start;

    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("Sequential took: %f, Simple Parallel took: %f, Complex Parallel took: %f\n", elapsed_sequential, elapsed_simple, elapsed_complex);
    printf("Speedup Simple Parallel: %f, Speedup Complex Parallel: %f\n", elapsed_sequential/elapsed_simple, elapsed_sequential/elapsed_complex);
    // std::cout << torch::equal(simple_tensor, sequential_tensor) << std::endl;
    std::cout << tm_simple(a, n) << std::endl;
    std::cout << tm_sequential(a, n) << std::endl;
    std::cout << tm(a, n) << std::endl;


}

int main(int argc, char *argv[]){
    if (argc < 2) {
        printf("invalid number of arguments");
        return 0;
    }
    // 0 - sequential, 1 - simple parallel, 2 - complex parallel
    // 0 - non-random, 1 - random
    int random = atoi(argv[1]);
    int nTen = atoi(argv[2]);

    double start, end;
       
    torch::Tensor tarr[nTen];

    if (random == 0){
        int szTen = atoi(argv[3]);
        int flag = atoi(argv[4]);

        for (int i = 0; i < nTen; ++i){
            tarr[i] = torch::randn({szTen, szTen});
        }
        torch::set_num_interop_threads(1);
        std::cout<< torch::get_num_threads() << '\n';

        switch(flag){
            case 0: 
                start = get_wtime();
                tm_sequential(tarr , nTen);
                end = get_wtime();
                break;
            case 1:
                start = get_wtime();
                tm_simple(tarr , nTen);
                end = get_wtime();
                break;
            case 2:
                start = get_wtime();
                tm(tarr, nTen);
                end = get_wtime();
                break;
        }   
        printf("Time: %f\n", end-start);
    } else {
        double seq_time;
        double simple_time;
        double complex_time;

        int rand_num_1 = rand() % szRAND + 1;
        int rand_num_2 = rand() % szRAND + 1;

        for (int i = 0; i < TRIALS; ++i){

            for (int i = 0; i < nTen; ++i){
                tarr[i] = torch::randn({rand_num_1, rand_num_2});
                rand_num_1 = rand_num_2;
                rand_num_2 = rand() % szRAND + 1;
            }
            start = get_wtime();
            tm_sequential(tarr, nTen);
            end = get_wtime();
            simple_time += end-start;

            for (int i = 0; i < nTen; ++i){
                tarr[i] = torch::randn({rand_num_1, rand_num_2});
                rand_num_1 = rand_num_2;
                rand_num_2 = rand() % szRAND + 1;
            }
            
            start = get_wtime();
            tm(tarr, nTen);
            end = get_wtime();
            complex_time += end-start;
            
            for (int i = 0; i < nTen; ++i){
                tarr[i] = torch::randn({rand_num_1, rand_num_2});
                rand_num_1 = rand_num_2;
                rand_num_2 = rand() % szRAND + 1;
            }
            start = get_wtime();
            tm_simple(tarr, nTen);
            end = get_wtime();
            seq_time += end-start;
        }
        printf("seq: %f, simple: %f, complex: %f\n", seq_time/TRIALS, simple_time/TRIALS, complex_time/TRIALS);
    }

    return 0;
}