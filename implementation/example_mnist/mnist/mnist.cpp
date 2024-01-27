#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>
#include <mpi.h>
#include <cfloat>

#include "model.h"
// #include "timer.h"
// Include this when running with OpenMP matrix multiplication, this is slower than PyTorch .backward().
// #include "parallel_multiply.h"

// Global parameters, some can be modified via arguments (see main function).

// Where to find the MNIST dataset.
const char* kDataRoot = "./data";

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
int64_t kTestBatchSize = 1000;

const int kMaxProbes = 200;

// The number of epochs to train.
int64_t kNumberOfEpochs;

// Number of steps we iterate through adversary generatoin
int64_t kAdversarySteps;

// The size of the gradient step (akin to learning rate).
// We use kStepSize = 1.0 for all our experiments.
float kStepSize;

// How many adversary steps are taken at test time
int64_t kTestAdversarySteps = 32;

// How often to check for a new model while generating an adversary
int64_t kCheckReceiveInterval = 10;

// After how many batches to log a new update with the loss value.
int64_t kLogInterval = 10;

// How many seconds to wait before checking if all the adversaries have finished being generated
const double kCheckFinishedInterval = 10.0;

int verbose = 1;

// Send the model weights from the weight-updating process to the 
//  adversary generating processes
MPI_Request** send_model_weights(const Net model) {
    torch::NoGradGuard no_grad;
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);    
    MPI_Request** requests = new MPI_Request*[size-1]; 
    for (int dest = 1; dest < size; dest++) {
        requests[dest - 1] = new MPI_Request[model.parameters().size()];
        int weight_num = 0;
        for (auto weight : model.parameters()) {
            MPI_Isend(weight.data_ptr(), weight.numel(), MPI_FLOAT, dest, weight_num, MPI_COMM_WORLD, &(requests[dest - 1][weight_num]));
            weight_num++;
        }
    }
    return requests;
}

// Fabian's answer to https://edstem.org/us/courses/20009/discussion/1431108, 
// suggests we do not want to use MPI_Ibcast since we are not really performing
// collective operations.
// Instead, we post receives, one for each layer/parameter of the network.
MPI_Request* receive_model_weights(Net &model, bool blocking) {
    torch::NoGradGuard no_grad;
    int weight_num = 0;
    MPI_Request* requests = new MPI_Request[model.parameters().size()];
    for (auto weight : model.parameters()) {
        if (blocking) {
            MPI_Recv(weight.data_ptr(), weight.numel(), MPI_FLOAT, 0, weight_num, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Irecv(weight.data_ptr(), weight.numel(), MPI_FLOAT, 0, weight_num, MPI_COMM_WORLD, &requests[weight_num]);
        }
        weight_num++;
    }
    if (blocking) {
        delete[] requests;
        return nullptr;
    } else {
        return requests;
    }
}

// This checks if the previous posted receive (of model weights) has finished, and 
// if so, posts a new one. However, it could be the case that 
// multiple sends from the model-generating process have been initiated 
// since the receive completed, in which case we get the latest 
// receive, using this method: 
// https://stackoverflow.com/questions/29764318/how-can-i-access-the-last-message-which-is-sent-to-a-processor-in-mpi
// This should update the requests pointer with the new request
// objects from the newest receive.
MPI_Request* check_receive_finished(
    Net &model,
    MPI_Request* requests
    ) {
    int received;
    MPI_Testall(model.parameters().size(), requests, &received, MPI_STATUSES_IGNORE);
    if (received == 0) {
        return requests;
    }

    int message_available;
    do {
        MPI_Iprobe(0, model.parameters().size()-1, MPI_COMM_WORLD, &message_available, MPI_STATUS_IGNORE);
        if (message_available) {
            // Post blocking receives until we receive latest model
            receive_model_weights(model, true);
        }
    } while (message_available);
    // Post asynchronous receive so the model can be updated while we are generating adversaries
    // The transfer part of the compute-transfer overlap!
    return receive_model_weights(model, false);
}

// Do the forward pass and backprop and get the gradient with respect to the input image
torch::Tensor get_input_grad(
    Net &model, 
    torch::Tensor input,
    torch::Tensor target
    ) {
        model.zero_grad();
        auto x = input.clone();
        x.requires_grad_(true);
        // std::vector<torch::Tensor> forward_pass = model.plain_forward(x);
        auto forward_pass = model.plain_forward(x);
        auto f = torch::sum(forward_pass)-2*torch::sum(torch::gather(forward_pass, 1, torch::unsqueeze(target, -1)));
        f.backward();
        return x.grad();
}

// Runs <steps> iterations of PGD (projected gradient descent)
torch::Tensor generate_adversary(
    Net &model, 
    torch::Tensor input, 
    torch::Tensor target,
    int steps, 
    int step_sz, 
    MPI_Request* &requests,
    float vmin = FLT_MIN, 
    float vmax = FLT_MAX
    ) {
        model.eval();
        torch::Tensor copied_input = input.clone();
        for (int i = 0; i < steps; ++i) {
            auto grad = get_input_grad(model, copied_input, target);
            copied_input += step_sz * grad ;
            copied_input = torch::clamp(copied_input, vmin, vmax);
            // We check if a new model has been sent every kCheckReceiveInterval steps
            // kCheckReceiveInterval=1 has the least staleness and best accuracy
            if (requests && i % kCheckReceiveInterval == 0) {
                requests = check_receive_finished(model, requests); 
            }
        }
        return copied_input;
    }

// Sequential training loop
template <typename DataLoader>
void train_sequential(
    size_t epoch,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {

    model.train();
    size_t batch_idx = 0;
    std::vector<double> adversary_times;
    std::vector<double> train_times;

    MPI_Request* temp = nullptr;

    auto training_start = std::chrono::high_resolution_clock::now();

    float total_loss = 0;

    // Loop through minibatches
    for (auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);

        auto adversary_start = std::chrono::high_resolution_clock::now();
        auto adversaries = generate_adversary(model, data, targets, kAdversarySteps, kStepSize, temp, -5, 5);
        auto adversary_end = std::chrono::high_resolution_clock::now();

        // Perform backprop on minibatch of adversaries
        model.train();
        optimizer.zero_grad();
        auto output = model.forward(adversaries);
        auto loss = torch::nll_loss(output, targets);
        AT_ASSERT(!std::isnan(loss.template item<float>()));
        loss.backward();
        optimizer.step();

        auto train_end = std::chrono::high_resolution_clock::now();
        double elapsed_adversary = std::chrono::duration<double>( adversary_end - adversary_start).count();
        double elapsed_training = std::chrono::duration<double>( train_end - adversary_end).count();

        train_times.push_back(elapsed_training);
        adversary_times.push_back(elapsed_adversary);

        total_loss += loss.template item<float>();
        
        if (++batch_idx % kLogInterval == 0 && verbose) {
            std::printf(
                "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f Time Elapsed: %.1fs\n",
                epoch,
                batch_idx * batch.data.size(0),
                dataset_size,
                loss.template item<float>(),
                std::chrono::duration<double>(train_end - training_start).count());
        }
    }

    auto epoch_end = std::chrono::high_resolution_clock::now();
    double total_epoch_time = std::chrono::duration<double>(epoch_end - training_start).count();

    std::printf(
        "\rTrain Epoch: %ld Loss: %.4f Time Elapsed: %.1fs\n",
        epoch, total_loss, total_epoch_time);

    double adversary_sum = std::accumulate(adversary_times.begin(), adversary_times.end(), 0.0);
    double adversary_mean = adversary_sum / adversary_times.size();

    double training_sum = std::accumulate(train_times.begin(), train_times.end(), 0.0);
    double training_mean = training_sum / train_times.size();

    printf("Average time to generate adversary: %f\n", adversary_mean);
    printf("Average time to train: %f\n", training_mean);
}


// Evaluates the performance of the model on the test dataset (sequentially),
// we typically use kTestAdversarySteps=32
template <typename DataLoader>
void test(
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
    model.eval();
    double test_loss = 0;
    int32_t correct = 0;
    MPI_Request* null_request = nullptr;
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device), targets = batch.target.to(device);
        auto adversaries = generate_adversary(model, data, targets, kTestAdversarySteps, kStepSize, null_request, -5, 5);
        {
            // Do the forward step
            torch::NoGradGuard no_grad;
            auto output = model.forward(adversaries);
            test_loss += torch::nll_loss(
                output,
                targets,
                /*weight=*/{},
                torch::Reduction::Sum)
                .template item<float>();
            auto pred = output.argmax(1);
            correct += pred.eq(targets).sum().template item<int64_t>();
        }
    }

    test_loss /= dataset_size;
    std::printf(
        "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
        test_loss,
        static_cast<double>(correct) / dataset_size);
}

// Sends an adversary from the adversary-generating process
// to the weight-training process
void send_adversary(
    const torch::Tensor adversaries, 
    const torch::Tensor targets
    ) {
    MPI_Request* requests = new MPI_Request[1];
    if (!adversaries.is_contiguous() or !targets.is_contiguous()) {
        throw std::logic_error("sending tensor is not contiguous");
    }
    MPI_Isend(adversaries.data_ptr(), adversaries.numel(), MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &requests[0]);
    MPI_Ssend(targets.data_ptr(), targets.numel(), MPI_LONG, 0, 101, MPI_COMM_WORLD);
}

// This loops through the dataset <kNumberOfEpochs> times and generates
//  adversaries and sends them to the weight-updating process.
template <typename DataLoader>
void adversary(
    Net &model,
    torch::Device device,
    DataLoader &data_loader
    ) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Request* requests = receive_model_weights(model, false);
    
    for (int epoch = 0; epoch < kNumberOfEpochs; epoch++) {
        for (auto& batch : data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            auto adversaries = generate_adversary(model, data, targets, kAdversarySteps, kStepSize, requests, -5, 5);
            send_adversary(adversaries, targets);
        }
        if (verbose) {
            std::cout << "Rank " << rank << " finished with epoch " << epoch << std::endl;
        }
    }
    int finished = 1;
    // Tag of 5000 is reserved for when processes are finished generating adversaries
    MPI_Ssend(&finished, 1, MPI_INT, 0, 5000, MPI_COMM_WORLD);
}

// Function for weight-updating process to receive adversaries
bool receive_adversaries(
    torch::Tensor &adversaries,
    torch::Tensor &targets,
    int &prev_source
    ) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int batch_available = 0;
    int count = 0;
    while (count < kMaxProbes) {
        // Instead of MPI_Waitsome, since we need to receive two messages from the source
        //  we implement a custom loop which probes to see if the two messages are available
        //  See Section 3 of report for fairness discussion, loop ensures some notion of fairness
        prev_source = (prev_source + 1) % (size - 1);
        int source = prev_source + 1;
        MPI_Iprobe(source, 101, MPI_COMM_WORLD, &batch_available, MPI_STATUS_IGNORE);
        if (batch_available) {
            MPI_Iprobe(source, 100, MPI_COMM_WORLD, &batch_available, MPI_STATUS_IGNORE);
            if (batch_available) {
                MPI_Recv(adversaries.data_ptr(), adversaries.numel(), MPI_FLOAT, source, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(targets.data_ptr(), targets.numel(), MPI_LONG, source, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                return true;
            }
        }
        count++;
    }
    return false;
}

// Updates the model weights based on the adversaries received
//  Very similar to backprop in train_sequential
void update_model_weights(
    Net &model, 
    torch::optim::Optimizer& optimizer,
    torch::Tensor &adversaries,
    torch::Tensor &targets,
    int &prev_source,
    int &epoch, 
    int &batch_idx,
    std::chrono::high_resolution_clock::time_point start_time,
    float &total_loss,
    double &time_elapsed
) {
    
    bool received = receive_adversaries(adversaries, targets, prev_source);
    if (!received) {
        return;
    }
    batch_idx++;
    optimizer.zero_grad();
    auto output = model.forward(adversaries);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();
    total_loss += loss.template item<float>();
    auto time = std::chrono::high_resolution_clock::now();
    time_elapsed = std::chrono::duration<double>(time - start_time).count();
    if (verbose && batch_idx % kLogInterval == 0) {
        std::printf(
            "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f Time Elapsed: %.1fs\n",
            epoch,
            batch_idx * adversaries.size(0),
            60000,
            loss.template item<float>(),
            time_elapsed);
        std::cout << std::flush;
    }
}

void train_parallel(
    Net &model,
    torch::Device device,
    torch::optim::Optimizer& optimizer
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto prev_check_time = start_time;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Post receives which will be matched when adv-generating processes finish
    MPI_Request done_requests[size-1];
    int receive_ptr[size - 1];
    for (int src = 1; src < size; src++) {
        // Tag of 5000 is reserved for the signal that a process has finished generating adversaries
        MPI_Irecv(&receive_ptr[src-1], 1, MPI_INT, src, 5000, MPI_COMM_WORLD, &done_requests[src-1]);
    }

    int epoch = 0;
    int batch_idx = 0;

    auto adversaries_options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(device).requires_grad(false);
    auto targets_options = torch::TensorOptions().dtype(torch::kInt64).layout(torch::kStrided).device(device).requires_grad(false);
    torch::Tensor adversaries = torch::empty({kTrainBatchSize, 1, 28, 28}, adversaries_options);
    torch::Tensor targets = torch::empty({kTrainBatchSize}, targets_options);
    int prev_source = 0;
    int done = 0;
    float total_loss = 0;
    double time_elapsed = 0;
    while (!done) {
        send_model_weights(model);
        update_model_weights(model, optimizer, adversaries, targets, prev_source, epoch, batch_idx, start_time, total_loss, time_elapsed);
        if (batch_idx / 933 > epoch) {
            epoch++;
            std::printf("\rTrain Epoch: %ld Loss: %.4f Time Elapsed: %.1fs\n",
                epoch, total_loss, time_elapsed);
            total_loss = 0;
            start_time = std::chrono::high_resolution_clock::now();
            std::cout << std::flush;
        }
        auto curr_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(curr_time - prev_check_time).count() > kCheckFinishedInterval) {
            // Check if all adversaries have finished being generated
            MPI_Testall(size - 1, done_requests, &done, MPI_STATUSES_IGNORE);
            prev_check_time = std::chrono::high_resolution_clock::now();
        }
    }
}

auto main(int argc, char* argv[]) -> int {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        MPI_Abort(MPI_COMM_WORLD, 123);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " numEpochs numAdversarySteps stepSize verbose" << std::endl;
        }
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        switch(i) {
        case 1:
            // The number of epochs to train.
            kNumberOfEpochs = atoi(argv[i]);
            break;
        case 2: // Number of steps we iterate through adversary generation
            kAdversarySteps = atoi(argv[i]);
            break;
        case 3:
            kStepSize = atof(argv[i]);
            break;
        case 4: // verbose must be 0 in order to run plotting after
            verbose = atoi(argv[i]);
            break;
        case 5: // How often to check for a new model while generating an adversary
            kCheckReceiveInterval = atoi(argv[i]);
            break;
        case 6: // After how many batches to log a new update with the loss value.
            kLogInterval = atoi(argv[i]);
            break;
        case 7:
            kTestAdversarySteps = atoi(argv[i]);
            break;
        default:
            break;
        }
    }

    torch::manual_seed(205);

    if (rank == 0 && verbose) {
        std::cout << "Torch OpenMP threads: " << torch::get_num_threads() << std::endl;
    }
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    }
    else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Net model;
    model.to(device);

    auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(
        kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();

    if (size > 1) {
        kTestBatchSize = 128;
    }
    auto test_loader =
        torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));

    if (size == 1) {
        // Run sequential code on a single process
        if (verbose) {
            std::cout << "Training sequentially." << std::endl;
        }
        std::printf("\rProcesses: %ld Epochs: %ld AdvSteps: %ld TestAdvSteps: %ld StepSize: %f\n", 
            size, kNumberOfEpochs, kAdversarySteps, kTestAdversarySteps, kStepSize);
        std::cout << std::flush;
        auto train_loader = 
            torch::data::make_data_loader(std::move(train_dataset), kTrainBatchSize);

        for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
            train_sequential(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        }
        test(model, device, *test_loader, test_dataset_size);
    } else {
        if (rank == 0) {
            if (verbose) {
                std::cout << "Training in parallel with " << size-1 << " adversary generating processes!" << std::endl;
            }
            std::printf("\rProcesses: %ld Epochs: %ld AdvSteps: %ld TestAdvSteps: %ld StepSize: %.1f ReceiveInterval: %ld\n", 
                size, kNumberOfEpochs, kAdversarySteps, kTestAdversarySteps, kStepSize, kCheckReceiveInterval);
            std::cout << std::flush;
            train_parallel(model, device, optimizer);
            test(model, device, *test_loader, test_dataset_size);
        } else {
            torch::data::samplers::DistributedRandomSampler sampler (train_dataset.size().value(), /*num_replicas=*/size-1, /*rank=*/rank-1);
            auto train_loader = 
                torch::data::make_data_loader(std::move(train_dataset), sampler, torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
            
            adversary(model, device, *train_loader);
        }
    }

    MPI_Finalize();

    return 0;
}
