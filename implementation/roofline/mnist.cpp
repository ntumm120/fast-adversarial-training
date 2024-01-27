#include <torch/torch.h>
#include "papi.h"
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>
#include <sys/time.h>

#define L1_SIZE_KB 32
#define L2_SIZE_KB 256
#define L3_SIZE_KB 40960

// Model name: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
// L1d cache:  32K
// L1i cache:  32K
// L2 cache:   256K
// L3 cache:   40960K
typedef float Real;


struct Net : torch::nn::Module {    
    Net()
        : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        fc1(8000, 50),
        fc2(50, 10) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        // register_module("conv2_drop", conv2_drop);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    std::vector<torch::Tensor> plain_forward(torch::Tensor x) {
        std::vector<torch::Tensor> forward_pass = {};
        x = conv1->forward(x);
        forward_pass.push_back(x);
        x = torch::relu(x);
        x = conv2->forward(x);
        forward_pass.push_back(x);
        x = torch::relu(x);
        x = x.view({ -1, 8000 });

        x = fc1->forward(x);
        forward_pass.push_back(x);
        x = torch::relu(x);

        // x = torch::dropout(x, /*p=*/0.1, /*training=*/is_training());
        x = fc2->forward(x);
        forward_pass.push_back(x);
        return forward_pass;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = x.view({ -1, 8000 });
        x = torch::relu(fc1->forward(x));
        // x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
        x = torch::log_softmax(fc2->forward(x), 1);
        return x;
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    // torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;

    torch::Tensor H1;
    torch::Tensor H2;
    torch::Tensor l1;
    torch::Tensor l2;
};

// timer
double get_wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6; // seconds
}

void test_compute_kernels(){
    Net model = Net();
    torch::Tensor test_tensor = torch::randn({1, 28, 28});
    torch::Tensor test_target = torch::tensor({4});

    auto output = model.forward(test_tensor);

    auto loss = torch::sum(output)-2*torch::sum(torch::gather(output, 1, torch::unsqueeze(test_target, -1)));
    loss.backward();


    auto conv1 = torch::nn::Conv2d(1, 10, /*kernel_size=*/5);
    auto ret_conv = conv1(test_tensor);

    torch::Tensor test_tensor2 = torch::randn({1, 8000});
    auto linear1 = torch::nn::Linear(8000, 50);

    auto ret_linear = linear1(test_tensor2);
}

int main(int argc, char* argv[]) {

    int n = 8000;
    int m = 50;
    int k = 10;
    int p = 28;

    int batch_size = 1;
    if (argc > 1) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
    }

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Net model = Net();
    model.to(device);
    torch::Tensor test_tensor = torch::randn({batch_size, p, p});
    torch::Tensor test_target = torch::tensor({4});
    torch::Tensor test_tensor2 = torch::randn({batch_size, n});
    auto conv1 = torch::nn::Conv2d(1, 10, /*kernel_size=*/5);
    auto linear1 = torch::nn::Linear(n, m);

    // Initialize PAPI
    int event_set = PAPI_NULL;
    int events[6] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_LST_INS, PAPI_L1_DCM, PAPI_SP_OPS, PAPI_VEC_SP};
    long long int counters[4];

    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&event_set);
    PAPI_add_events(event_set, events, 4);

    // warm up
    // linear1(test_tensor2);
    // conv1(test_tensor);
    // auto output = model.forward(test_tensor);
    // auto loss = torch::sum(output)-2*torch::sum(torch::gather(output, 1, torch::unsqueeze(test_target, -1)));
    // loss.backward();

    // start PAPI measurement
    PAPI_start(event_set);
    const long long int t0 = PAPI_get_real_nsec();

    linear1(test_tensor2);
    // conv1(test_tensor);
    // model.forward(test_tensor);
    // loss.backward();

    const long long int t1 = PAPI_get_real_nsec();
    PAPI_stop(event_set, counters);

    // clang-format off
    const long long total_cycles = counters[0];       // cpu cycles
    const long long total_instructions = counters[1]; // any
    const long long total_load_stores = counters[2];  // number of such instructions
    const long long total_l1d_misses = counters[3];   // number of access request to cache line
    const long long flops_ss = counters[4];
    const long long flops_sp = counters[5];
    // clang-format on  
    
    const size_t flops = 799950;
    const size_t mem_ops = 799950+100;
    const double twall = (static_cast<double>(t1) - t0) * 1.0e-9; // seconds
    const double IPC = static_cast<double>(total_instructions) / total_cycles;
    const double OI =
        static_cast<double>(flops) / (total_load_stores * sizeof(Real));
    const double OI_theory =
        static_cast<double>(flops) / (mem_ops * sizeof(Real));
    const double float_perf = flops / twall * 1.0e-9; // Gflop/s

    // std::cout << "Result:                       " << res << '\n';
    std::cout << "Total cycles:                 " << total_cycles << '\n';
    std::cout << "Total instructions:           " << total_instructions << '\n';
    std::cout << "Instructions per cycle (IPC): " << IPC << '\n';
    std::cout << "L1 cache size:                " << L1_SIZE_KB << " KB\n";
    std::cout << "L2 cache size:                " << L2_SIZE_KB << " KB\n";
    std::cout << "L3 cache size:                " << L3_SIZE_KB << " KB\n";
    // std::cout << "Total problem size:           "
    //           << 3 * n * n * sizeof(Real) / 1024 << " KB\n";
    std::cout << "Total flops:                  " << flops << '\n';
    std::cout << "Total flops ss:               " << flops_ss << '\n';
    std::cout << "Total flops sp:               " << flops_sp << '\n';
    std::cout << "Total L1 data misses:         " << total_l1d_misses << '\n';
    std::cout << "Total load/store:             " << total_load_stores
              << " (expected: " << mem_ops << ")\n";
    std::cout << "Operational intensity:        " << std::scientific << OI
              << " (expected: " << OI_theory << ")\n";
    std::cout << "Performance [Gflop/s]:        " << float_perf << '\n';
    std::cout << "Wall-time   [micro-seconds]:  " << twall * 1.0e6 << '\n';
    return 0;
}