#include <iostream>
#include <stdlib.h>

#ifdef KERNEL1
#include "kernel1.cuh"
#endif
#ifdef KERNEL2
#include "kernel2.cuh"
#endif
#ifdef KERNEL3
#include "kernel3.cuh"
#endif

#include "support.h"

int main(int argc, char *argv[])
{
    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    std::cout << "\nSetting up the problem..." << std::endl;
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1)
    {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    }
    else if (argc == 2)
    {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    }
    else if (argc == 4)
    {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    }
    else
    {
        std::cout << "    Invalid input parameters!" << std::endl;
        std::cout << "    Usage: ./sgemm-tiled                # All matrices are 1000 x 1000" << std::endl;
        std::cout << "    Usage: ./sgemm-tiled <m>            # All matrices are m x m" << std::endl;
        std::cout << "    Usage: ./sgemm-tiled <m> <k> <n>    # A: m x k, B: k x n, C: m x n" << std::endl;
        exit(EXIT_FAILURE);
    }

    A_sz = matArow * matAcol;
    B_sz = matBrow * matBcol;
    C_sz = matArow * matBcol;

    cudaMallocHost((void **)&A_h, sizeof(float) * A_sz);
    for (unsigned int i = 0; i < A_sz; i++)
    {
        A_h[i] = (rand() % 100) / 100.00;
    }

    cudaMallocHost((void **)&B_h, sizeof(float) * B_sz);
    for (unsigned int i = 0; i < B_sz; i++)
    {
        B_h[i] = (rand() % 100) / 100.00;
    }

    cudaMallocHost((void **)&C_h, sizeof(float) * C_sz);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << "s" << std::endl;
    std::cout << "    A: matArow x matAcol" << std::endl;
    std::cout << "    B: matBrow x matBcol" << std::endl;
    std::cout << "    C: matArow x matBcol" << std::endl;

    // Launch kernel using msplitm ---------------------------
    std::cout << "Launching kernel..." << std::endl;
    startTime(&timer);
    msplitm('N', 'N', matArow, matBcol, matBrow, 1.0f, A_h, matArow, B_h, matBrow, 0.0f, C_h, matBrow);

    cuda_ret = cudaDeviceSynchronize();
    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    // Verify correctness -----------------------------------------------------
    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);

    // Free memory ------------------------------------------------------------

    cudaFreeHost(A_h);
    cudaFreeHost(B_h);
    cudaFreeHost(C_h);

    return 0;
}
