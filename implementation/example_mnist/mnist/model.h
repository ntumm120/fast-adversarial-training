#include <torch/torch.h>
#include <iostream>

using namespace torch::indexing;

torch::Tensor convmatrix2d(torch::Tensor kernel, torch::Tensor image_shape){
    // kernel: (out_channels, in_channels, kernel_height, kernel_width)
    // image: (in_channels, image_height, image_width)

    torch::Tensor result_dims = image_shape.index({Slice(1, None)}) - (torch::tensor(kernel.sizes())).index({Slice(2, None)}) + 1;
    torch::Tensor m = torch::zeros({kernel.sizes()[0], result_dims.index({0}).item<int>(), result_dims.index({1}).item<int>(), image_shape.index({0}).item<int>(), image_shape.index({1}).item<int>(), image_shape.index({2}).item<int>()});
    
    for (int i = 0; i < m.sizes()[1]; i++){
        for (int j = 0; j < m.sizes()[2]; j++){
            m.index({Slice(0, None), i, j, Slice(0, None), Slice(i,i+kernel.sizes()[2]), Slice(j,j+kernel.sizes()[3])}) = kernel;
        }
    }
    return m.reshape({-1, image_shape.index({0}).item<int>() * image_shape.index({1}).item<int>() * image_shape.index({2}).item<int>()});

}

torch::Tensor relu_grad(torch::Tensor M, torch::Tensor input){
    torch::Tensor mask = torch::where(input>=0, 1, 0).reshape({M.sizes()[0], M.sizes()[1], 1});
    return torch::mul(mask, M);    
}

torch::Tensor softmax_grad(torch::Tensor s){
    return torch::diag_embed(s) - torch::einsum("br,bs->brs", {s, s});    
}

torch::Tensor loss_grad(torch::Tensor targets){
    return ((targets * 2 - 1) * -1).unsqueeze(1);   
}

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

    // std::vector<torch::Tensor> plain_forward(torch::Tensor x) {
    torch::Tensor plain_forward(torch::Tensor x) {
        // std::vector<torch::Tensor> forward_pass = {};
        x = conv1->forward(x);
        // forward_pass.push_back(x);
        x = torch::relu(x);
        x = conv2->forward(x);
        // forward_pass.push_back(x);
        x = torch::relu(x);
        x = x.view({ -1, 8000 });

        x = fc1->forward(x);
        // forward_pass.push_back(x);
        x = torch::relu(x);

        // x = torch::dropout(x, /*p=*/0.1, /*training=*/is_training());
        x = fc2->forward(x);
        // forward_pass.push_back(x);
        return x; // forward_pass;
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