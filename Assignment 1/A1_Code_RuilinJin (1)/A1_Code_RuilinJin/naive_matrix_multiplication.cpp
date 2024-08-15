#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>  // Include this header for timing

int main()
{
    const int N = 2048; // Side length of the square matrices
    std::vector<std::vector<double> > matrixA(N, std::vector<double>(N));
    std::vector<std::vector<double> > matrixB(N, std::vector<double>(N));
    std::vector<std::vector<double> > result(N, std::vector<double>(N, 0.0));

    // Read matrix A from a file
    std::ifstream fileA("matrixA.txt");
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            fileA >> matrixA[i][j];
        }
    }
    fileA.close();

    // Read matrix B from a file
    std::ifstream fileB("matrixB.txt");
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            fileB >> matrixB[i][j];
        }
    }
    fileB.close();

    // Start the timer right before the matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();

    // Matrix multiplication
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < N; ++k)
            {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    // Stop the timer right after the matrix multiplication
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time taken by matrix multiplication: " << duration.count() << " microseconds" << std::endl;

    // Write the resulting matrix to a file
    std::ofstream outFile("resultMatrix.txt");
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            outFile << result[i][j] << " ";
        }
        outFile << std::endl; // New line for each row
    }
    outFile.close(); 

    std::cout << "Matrix multiplication results have been written to 'resultMatrix.txt'." << std::endl;

    return 0;
}
