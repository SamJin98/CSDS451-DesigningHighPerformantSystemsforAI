#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cblas.h>

const int MATRIX_SIZE = 2048;

int main() {
  std::vector<std::vector<double>> matrixA(MATRIX_SIZE, std::vector<double>(MATRIX_SIZE));
  std::vector<std::vector<double>> matrixB(MATRIX_SIZE, std::vector<double>(MATRIX_SIZE));
  std::vector<std::vector<double>> result(MATRIX_SIZE, std::vector<double>(MATRIX_SIZE));

  std::ifstream fileA("matrixA.txt"), fileB("matrixB.txt");

  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      fileA >> matrixA[i][j];
    }
  }

  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      fileB >> matrixB[i][j];
    }
  }

  auto start = std::chrono::high_resolution_clock::now();

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1.0, &matrixA[0][0], MATRIX_SIZE, &matrixB[0][0], MATRIX_SIZE, 0.0, &result[0][0], MATRIX_SIZE);

  auto stop = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by multiplication: " << duration.count() << " microseconds" << std::endl;

  // To calculate FLOPS:
  double FLOPS = (2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE - MATRIX_SIZE * MATRIX_SIZE) / (duration.count() * 1e-6);
  std::cout << "FLOPS: " << FLOPS << std::endl;

  // Save the output matrix if needed
  std::ofstream outputFile("outputMatrix.txt");
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      outputFile << result[i][j] << " ";
    }
    outputFile << "\n";
  }

  return 0;
}
