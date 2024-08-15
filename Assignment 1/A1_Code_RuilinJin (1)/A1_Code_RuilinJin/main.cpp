#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <iomanip>

const int MATRIX_SIZE = 2048;

int main() {
  std::ofstream fileA("matrixA.txt"), fileB("matrixB.txt");
  std::mt19937 gen(0);  // Using a fixed seed for reproducibility
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      double value = dis(gen);
      fileA << std::fixed << std::setprecision(3) << value << " ";  // Use fixed and setprecision here
    }
    fileA << "\n";
  }

  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      double value = dis(gen);
      fileB << std::fixed << std::setprecision(3) << value << " ";  // And here as well
    }
    fileB << "\n";
  }

  fileA.close();
  fileB.close();

  return 0;
}
