#include <cudnn.h>
#include <iostream>
int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    std::cout << "cuDNN version: " << CUDNN_MAJOR << "." << CUDNN_MINOR << "." << CUDNN_PATCHLEVEL << std::endl;
    cudnnDestroy(cudnn);
    return 0;
}