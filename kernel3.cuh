#include <iostream>
#include <cublas_v2.h>

#include "common.h"

const int num_submatrix = 2;
const int numStreams = 2;

cudaStream_t streams[numStreams];
float *b[numStreams];
float *a[numStreams];
float *c[numStreams];
float *a_h[numStreams];
float *c_h[numStreams];
cublasHandle_t handles[numStreams];

void msplitm(char transa, char transb, unsigned long long m, unsigned long long n, unsigned long long k, float alpha, float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    std::cout << "entering msplitm" << std::endl;
    float *A_d;
    float *B_d;
    float *C_d;
    unsigned long long A_sz = m * k;
    unsigned long long B_sz = n * k;
    unsigned long long C_sz = m * n;
    unsigned long long MAX = (unsigned long long)m * (unsigned long long)n / num_submatrix;

    MAX -= MAX % k;
    std::cout << "MAX: " << MAX << std::endl;
    std::cout << "B_sz: " << B_sz << std::endl;
    unsigned long long numSubMatrixB = B_sz / MAX;
    std::cout << "SubmatriciesB: " << numSubMatrixB << std::endl;
    unsigned long long SMB_sz = B_sz / numSubMatrixB;
    std::cout << "SMB_sz: " << SMB_sz << std::endl;
    unsigned long long subCols = B_sz / (numSubMatrixB * k);
    std::cout << "subCols: " << subCols << std::endl;

    unsigned long long numSubMatrixA = A_sz / MAX;
    unsigned long long SMA_sz = A_sz / numSubMatrixA;
    unsigned long long subRows = A_sz / (numSubMatrixA * k);
    std::cout << "subrows: " << subRows << std::endl;
    std::cout << "SMA_sz: " << SMA_sz << std::endl;
    std::cout << "submatriciesA: " << numSubMatrixA << std::endl;
    unsigned long long overflowA = m % subRows;
    unsigned long long overflowB = n % subCols;
    std::cout << "overflowB: " << overflowB << std::endl;
    std::cout << "overflowA: " << overflowA << std::endl;
    for (int i = 0; i < numStreams; ++i)
    {
        cudaSetDevice(i);
        cublasCreate(&handles[i]);
        cudaStreamCreate(&streams[i]);
        cudaMalloc((void **)&b[i], sizeof(float) * subCols * k);
        cudaMalloc((void **)&a[i], sizeof(float) * subRows * k);
        cudaMalloc((void **)&c[i], sizeof(float) * subCols * subRows);
        cudaMallocHost((void **)&a_h[i], sizeof(float) * subRows * k);
        cudaMallocHost((void **)&c_h[i], sizeof(float) * subCols * subRows);
    }

    for (unsigned long long i = 0; i < numSubMatrixB + 1; ++i)
    {
        int count = 0;
        if (overflowB == 0 && i == numSubMatrixB)
        {
            break;
        }

        int copynumB = i == numSubMatrixB ? overflowB : subCols;
        for (int j = 0; j < numStreams; ++j)
        {
            cudaSetDevice(j);
            if (i == numSubMatrixB)
            {
                cudaMemsetAsync(a, 0, sizeof(float) * k * subCols, streams[j]);
            }
            cudaMemcpy2DAsync(b[j], subCols * sizeof(float), B + (i * subCols), n * sizeof(float),
                              copynumB * sizeof(float), k, cudaMemcpyHostToDevice, streams[j]);
        }
        unsigned long long y = 0;
        int streamsActive = 0;
        while (y < numSubMatrixA + 1)
        {
            if (overflowA == 0 && y == numSubMatrixA)
            {
                break;
            }
            int copynumA = y == numSubMatrixA ? overflowA : subRows;
            cudaSetDevice(y % numStreams);
            if (y == numSubMatrixA)
            {
                cudaMemsetAsync(a, 0, sizeof(float) * k * subRows, streams[y % numStreams]);
            }
            cudaMemcpy2DAsync(a[y % numStreams], k * sizeof(float), A + (k * y * subRows), k * sizeof(float),
                              k * sizeof(float), copynumA, cudaMemcpyHostToDevice, streams[y % numStreams]);

            std::cout << "sending multiply " << y << ", " << i << " to stream " <<
                y % numStreams << std::endl;
            doMultiply2MatricesStreaming(subRows, k, a[y % numStreams], k, subCols, b[y % numStreams], c[y % numStreams], streams[y % numStreams], handles[y % numStreams], alpha);
            cudaMemcpyAsync(c_h[y % numStreams], c[y % numStreams], sizeof(float) * subRows * subCols, cudaMemcpyDeviceToHost, streams[y % numStreams]);

            streamsActive++;
            if (y % numStreams == numStreams - 1)
            {
                for (int s = 0; s < numStreams; ++s)
                {
                    cudaStreamSynchronize(streams[s]);
                    int currWork = count * numStreams + s;
                    // TODO: We can probably do a direct copy from the device to the appropriate output location on the host
                    // But we need to handle the beta term on the GPU
                    if (i == numSubMatrixB && currWork == numSubMatrixA)
                    {
                        copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
                    }
                    else if (i == numSubMatrixB)
                    {
                        copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
                    }
                    else if (currWork == numSubMatrixA)
                    {
                        copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
                    }
                    else
                    {
                        copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
                    }
                    streamsActive--;
                }
                ++count;
            }
            ++y;
        }

        for (int s = 0; s < streamsActive; ++s)
        {
            cudaStreamSynchronize(streams[s]);
            int currWork = count * numStreams + s;
            if (i == numSubMatrixB && currWork == numSubMatrixA)
            {
                copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, overflowA, overflowB, beta);
            }
            else if (i == numSubMatrixB)
            {
                copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, overflowB, beta);
            }
            else if (currWork == numSubMatrixA)
            {
                copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, overflowA, 0, beta);
            }
            else
            {
                copyElements(C, c_h[s], subRows, subCols, m, n, currWork, i, 0, 0, beta);
            }
        }
    }

    for (int i = 0; i < numStreams; ++i)
    {
        cudaSetDevice(i);
        cudaFree(a[i]);
        cudaFree(c[i]);
        cudaFreeHost(a_h[i]);
        cudaFreeHost(c_h[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(b);
}
