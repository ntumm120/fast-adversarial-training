#include <torch/torch.h>
#include <omp.h>
#include "timer.h"

#define SZ 100

torch::Tensor tm(torch::Tensor* tensors, size_t n);

torch::Tensor tm_simple(torch::Tensor* tensors, size_t n);

torch::Tensor tm_sequential(torch::Tensor* tensors, size_t n);

// 0. copy array for preprocessing
// 1. queue of tasks
// 2. array for next pointers
// 3. array for locks on elements of torch array
// 4. while pointer[0] != -1: (only happens after 0 is done multiplied))
//      a. lock task from queue
//          i. if the pointer is locked, iterate through the next task
//          ii. if pointer is not locked, take the task of the queue, update pointer of pointer
//               (both left -> next, right -> -1), multiply tensors, replace task tensor with other
// 5. after while, return the first element

// takes in a list of pointers to the tensor, and then returns the tensor
torch::Tensor tm(torch::Tensor* tensors, size_t n){
    
    torch::Tensor tensor_arr[n];
    std::queue<int> tasks;
    int next_tensor[n];
    omp_lock_t lock[n];
    omp_lock_t task_lock;
    std::atomic<bool> finished;

    omp_init_lock(&task_lock);

    for (int i=0; i<n; i++){
        tensor_arr[i] = tensors[i];
        omp_init_lock(&(lock[i]));
        next_tensor[i] = i+1;
        if (i == n-1){
            next_tensor[i] = -1;
        }
        if (i % 2 == 0){
            tasks.push(i);
        }
    }
    printf("nThreads: %d\n", omp_get_max_threads());
    #pragma omp parallel 
    {   
        torch::set_num_threads(1);
        while(!finished){
            omp_set_lock(&task_lock);
            //if empty continue
            if (tasks.empty()){
                omp_unset_lock(&task_lock);
                continue;
            }
            int task = tasks.front();
            tasks.pop();
            // locked first task
            if (omp_test_lock(&(lock[task]))){
                int next_task = next_tensor[task];
                if (next_task > -1){
                    //locked second task
                    if (omp_test_lock(&(lock[next_task]))){
                        tasks.push(task);
                        omp_unset_lock(&task_lock);
                        //multiply
                        tensor_arr[task] = torch::matmul(tensor_arr[task], tensor_arr[next_task]);
                        //update pointers
                        next_tensor[task] = next_tensor[next_task];
                        next_tensor[next_task] = -1;
                        omp_unset_lock(&(lock[task]));
                        omp_unset_lock(&(lock[next_task]));
                    //didn't lock second task
                    } else {
                        tasks.push(task);
                        omp_unset_lock(&(lock[task]));
                        omp_unset_lock(&task_lock);
                    }
                } else {
                    // task points to -1 (nothing left to do)
                    if (task == 0){
                        finished = true;
                    }
                    omp_unset_lock(&(lock[task]));
                    omp_unset_lock(&task_lock);
                }
                omp_unset_lock(&(lock[task]));
            // didn't lock first task
            } else {
                tasks.push(task);
                omp_unset_lock(&task_lock);
            }
        }
    }
    
    return tensor_arr[0];
}

torch::Tensor tm_simple(torch::Tensor* tensors, size_t n){

    int nThreads = std::min(omp_get_max_threads(), (int) n/2);
    int block_size = int(n)/nThreads;
    int rem = int(n)%nThreads;
    torch::Tensor t[nThreads];
    printf("nThreads: %d\n", nThreads);
    const double start = get_wtime();
    #pragma omp parallel num_threads(nThreads)
    {       
        torch::set_num_threads(1);
        int tid = omp_get_thread_num();

        int start = tid*block_size;
        int end = (tid+1)*block_size;
        if (tid < rem){
            if (tid == 0){
                end += 1;
            } else {
                start += tid;
                end += tid+1;
            }
        } else {
            start += rem;
            end += rem;
        }

        torch::Tensor t_arr[end-start];
        // printf("Start: %d, End: %d\n", start, end);

        for (int i = 0; i < end-start; ++i){
            t_arr[i] = tensors[start+i];
        }
        const double mstart = get_wtime();
        for (int i = 1; i < end-start; ++i){
            t_arr[0] = torch::matmul(t_arr[0], t_arr[i]);
        }
        const double mend = get_wtime();
        const double elapsed_m = mend-mstart;
        // #pragma omp critical
        // {
        //     printf("Id: %d, For Loop: %f\n", tid, elapsed_m);
        // }
        t[tid] = t_arr[0];
    }

    const double end = get_wtime();

    const double elapsed_parallel = end-start;
    const double sstart = get_wtime();
    for (int i = 1; i < nThreads; ++i){
        t[0] = torch::matmul(t[0], t[i]);
    }
    const double send = get_wtime();
    const double elapsed_sequential = send - sstart;
    // printf("Parallel region took: %f, Sequential region took: %f\n", elapsed_parallel, elapsed_sequential);
    return t[0];
}

torch::Tensor tm_sequential(torch::Tensor* tensors, size_t n){
    torch::Tensor tensor_arr[n];

    for (int i = 0; i < n; ++i){
        tensor_arr[i] = tensors[i];
    }
    torch::Tensor t = tensor_arr[0];

    const double sstart = get_wtime();
    // torch::set_num_threads(1);
    for (size_t i = 1; i < n ; ++i){
        t = torch::matmul(t, tensor_arr[i]);
    }

    const double send = get_wtime();
    const double elapsed_sequential = send - sstart;
    // printf("All sequential region took: %f\n", elapsed_sequential);
    return t;
}

void test_matrix_multiplication(){
    // float arr[SZ*SZ];
    // float arr2[SZ*SZ];
    // float arr3[SZ*SZ];
    // float arr4[SZ*SZ];

    // for (int i = 0; i < SZ*SZ; ++i) {
    //     arr[i] = float(i+2);
    //     arr2[i] = float(i+1);
    //     arr3[i] = float(i+3);
    //     arr4[4] = float(i+0);
    // }
    torch::Tensor timed_tensor1 = torch::randn({SZ, SZ});
    torch::Tensor timed_tensor2 = torch::randn({SZ, SZ});

    const double fstart = get_wtime();

    torch::Tensor timed_tensor = torch::matmul(timed_tensor1, timed_tensor2);

    const double fend = get_wtime();

    printf("Sequential timed: %f\n", fend-fstart);

    #pragma omp parallel 
    {
	    torch::Tensor timed_tensor3 = torch::randn({SZ, SZ});
	    torch::Tensor timed_tensor4 = torch::randn({SZ, SZ});
        int tid = omp_get_thread_num();
        std::cout << "Torch OpenMP threads: " << torch::get_num_threads() << std::endl;
        const double mystart = get_wtime();

        torch::Tensor test_tensor = torch::matmul(timed_tensor3, timed_tensor4);

        const double myend = get_wtime();

        #pragma omp critical
        {
            printf("ID: %d, Sequential timed: %f\n", tid, myend-mystart);
        }
    }

    torch::set_num_threads(1);
    torch::Tensor timed_tensor5 = torch::randn({SZ, SZ});
    torch::Tensor timed_tensor6 = torch::randn({SZ, SZ});

    const double start = get_wtime();

    torch::Tensor timed_tensor_1 = torch::matmul(timed_tensor5, timed_tensor6);

    const double end = get_wtime();

    printf("Sequential timed: %f\n", end-start);
}