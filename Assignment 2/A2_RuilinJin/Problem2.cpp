#include <iostream>
#include <mkl.h>
#include <vector>
#include <chrono>

constexpr int N = 2048;

int main()
{
    // Allocate matrices
    double *A = (double *)mkl_malloc(N * N * sizeof(double), 64);
    double *B = (double *)mkl_malloc(N * N * sizeof(double), 64);
    double *C = (double *)mkl_malloc(N * N * sizeof(double), 64);

    // Fill matrices A and B with random numbers
    for (int i = 0; i < N * N; i++)
    {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    // Parameters for GEMM
    char trans = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    int lda = N;
    int ldb = N;
    int ldc = N;

    // Time and run the GEMM routine
    auto start = std::chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, lda, B, ldb, beta, C, ldc);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Time taken by MKL's GEMM: " << duration << " ms" << std::endl;

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
