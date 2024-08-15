#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

using namespace cl::sycl;

constexpr int N = 2048;

// Naive matrix multiplication kernel
class MatrixMulKernel;

int main()
{
    // Create two random matrices
    std::vector<float> matA(N * N);
    std::vector<float> matB(N * N);
    std::vector<float> matC(N * N, 0.0f); // Output matrix

    // Random number generation
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < N * N; i++)
    {
        matA[i] = dis(gen);
        matB[i] = dis(gen);
    }

    // Set up the DPC++ queue and device
    queue q;

    {
        // Create buffers
        buffer<float, 1> bufA(matA.data(), matA.size());
        buffer<float, 1> bufB(matB.data(), matB.size());
        buffer<float, 1> bufC(matC.data(), matC.size());

        // Submit the kernel for execution
        q.submit([&](handler &h)
                 {
            auto accA = bufA.get_access<access::mode::read>(h);
            auto accB = bufB.get_access<access::mode::read>(h);
            auto accC = bufC.get_access<access::mode::write>(h);

            h.parallel_for<MatrixMulKernel>(
                range<2>{N, N}, [=](id<2> idx) {
                    int global_x = idx[0];
                    int global_y = idx[1];
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += accA[global_y * N + k] * accB[k * N + global_x];
                    }
                    accC[global_y * N + global_x] = sum;
                }); });
    }

    // (Optional) Print a small portion of the result for verification
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            std::cout << matC[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
