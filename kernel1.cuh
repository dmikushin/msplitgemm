#include <iostream>
#include <cublas_v2.h>

#include "common.h"
#include "support.h"

const int num_submatrix = 8;

void msplitm(char transa, char transb, unsigned long long m, unsigned long long n, unsigned long long k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
	std::cout << "entering msplitm" << std::endl;
	unsigned long long A_sz = m * k;
	unsigned long long B_sz = n * k;
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
	for (unsigned long long i = 0; i < numSubMatrixB + 1; ++i)
	{
		if (overflowB == 0 && i == numSubMatrixB)
		{
			break;
		}
		float *b = 0;
		float *temp3 = (float *)malloc(sizeof(float) * subCols * k);
		for (int j = 0; j < k; ++j)
		{
			for (int x = 0; x < subCols; ++x)
			{
				if (i * subCols + x < n)
				{
					temp3[j * subCols + x] = B[j * n + (i * subCols + x)];
				}
				else
				{
					temp3[j * subCols + x] = 0;
				}
			}
		}
		CUDA_ERR_CHECK(cudaMalloc((void **)&b, sizeof(float) * subCols * k));
		CUDA_ERR_CHECK(cudaMemcpy(b, temp3, sizeof(float) * subCols * k, cudaMemcpyHostToDevice));
		free(temp3);
		for (unsigned long long y = 0; y < numSubMatrixA + 1; ++y)
		{
			if (overflowA == 0 && y == numSubMatrixA)
			{
				break;
			}
			float *temp = (float *)malloc(sizeof(float) * subRows * k);
			for (int j = 0; j < subRows; ++j)
			{
				for (int x = 0; x < k; ++x)
				{
					if (y * subRows + j < m)
					{
						temp[j * k + x] = A[y * subRows * k + j * k + x];
					}
					else
					{
						temp[j * k + x] = 0;
					}
				}
			}
			float *a = 0;
			float *c = 0;
			CUDA_ERR_CHECK(cudaMalloc((void **)&a, sizeof(float) * subRows * k));
			CUDA_ERR_CHECK(cudaMalloc((void **)&c, sizeof(float) * subCols * subRows));
			CUDA_ERR_CHECK(cudaMemcpy(a, temp, sizeof(float) * subRows * k, cudaMemcpyHostToDevice));
			doMultiply2Matrices(subRows, k, a, k, subCols, b, c, alpha);
			CUDA_ERR_CHECK(cudaMemcpy(temp, c, sizeof(float) * subRows * subCols, cudaMemcpyDeviceToHost));
			if (i == numSubMatrixB && y == numSubMatrixA)
			{
				copyElements(C, temp, subRows, subCols, m, n, y, i, overflowA, overflowB, beta);
			}
			else if (i == numSubMatrixB)
			{
				copyElements(C, temp, subRows, subCols, m, n, y, i, 0, overflowB, beta);
			}
			else if (y == numSubMatrixA)
			{
				copyElements(C, temp, subRows, subCols, m, n, y, i, overflowA, 0, beta);
			}
			else
			{
				copyElements(C, temp, subRows, subCols, m, n, y, i, 0, 0, beta);
			}
			free(temp);
			CUDA_ERR_CHECK(cudaFree(a));
			CUDA_ERR_CHECK(cudaFree(c));
		}

		CUDA_ERR_CHECK(cudaFree(b));
	}
}
