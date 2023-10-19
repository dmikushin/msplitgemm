#include <iostream>
#include <cublas_v2.h>

#include "common.h"

const int num_submatrix = 8;

void copyElements(float *out, float *entry, unsigned long long eRows, unsigned long long eCols, unsigned long long oRows, unsigned long long oCols, unsigned long long x, unsigned long long y,
				  unsigned long long ofA, unsigned long long ofB)
{
	unsigned long long counterRows = eRows;
	unsigned long long counterCols = eCols;
	if (ofA)
	{
		counterRows = ofA;
	}
	if (ofB)
	{
		counterCols = ofB;
	}
	for (unsigned long long i = 0; i < counterRows; ++i)
	{
		for (unsigned long long j = 0; j < counterCols; ++j)
		{
			out[x * eRows * oCols + (i * oCols) + (y * eCols + j)] = entry[i * eCols + j];
		}
	}
}

void msplitm(char transa, char transb, unsigned long long m, unsigned long long n, unsigned long long k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
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
	float **B_split = (float **)malloc(sizeof(float *) * (numSubMatrixB + 1));
	for (int i = 0; i < numSubMatrixB + 1; ++i)
	{
		float *temp = (float *)malloc(sizeof(float) * subCols * k);
		for (int j = 0; j < k; ++j)
		{
			for (int x = 0; x < subCols; ++x)
			{
				if (i * subCols + x < n)
				{
					temp[j * subCols + x] = B[j * n + (i * subCols + x)];
				}
				else
				{
					temp[j * subCols + x] = 0;
				}
			}
		}
		cudaMalloc((void **)&B_split[i], sizeof(float) * subCols * k);
		cudaMemcpy(B_split[i], temp, sizeof(float) * subCols * k, cudaMemcpyHostToDevice);
		free(temp);
	}
	for (unsigned long long i = 0; i < numSubMatrixA + 1; ++i)
	{
		if (overflowA == 0 && i == numSubMatrixA)
		{
			break;
		}
		float *temp = (float *)malloc(sizeof(float) * subRows * k);
		for (int j = 0; j < subRows; ++j)
		{
			for (int x = 0; x < k; ++x)
			{
				if (i * subRows + j < m)
				{
					temp[j * k + x] = A[i * subRows * k + j * k + x];
				}
				else
				{
					temp[j * k + x] = 0;
				}
			}
		}
		float *temp2 = 0;
		float *temp3 = 0;
		cudaMalloc((void **)&temp2, sizeof(float) * subRows * k);
		cudaMalloc((void **)&temp3, sizeof(float) * subCols * subRows);
		cudaMemcpy(temp2, temp, sizeof(float) * subRows * k, cudaMemcpyHostToDevice);
		free(temp);

		std::cout << "Running multiply for row group " << i << std::endl;
		temp = (float *)malloc(sizeof(float) * subRows * subCols);
		for (int x = 0; x < numSubMatrixB + 1; ++x)
		{
			if (overflowB == 0 && x == numSubMatrixB)
			{
				break;
			}
			doMultiply2Matrices(subRows, k, temp2, k, subCols, B_split[x], temp3, alpha);

			cudaMemcpy(temp, temp3, sizeof(float) * subRows * subCols, cudaMemcpyDeviceToHost);

			if (x == numSubMatrixB && i == numSubMatrixA)
			{
				copyElements(C, temp, subRows, subCols, m, n, i, x, overflowA, overflowB, beta);
			}
			else if (x == numSubMatrixB)
			{
				copyElements(C, temp, subRows, subCols, m, n, i, x, 0, overflowB, beta);
			}
			else if (i == numSubMatrixA)
			{
				copyElements(C, temp, subRows, subCols, m, n, i, x, overflowA, 0, beta);
			}
			else
			{
				copyElements(C, temp, subRows, subCols, m, n, i, x, 0, 0, beta);
			}
		}

		cudaFree(temp2);
		cudaFree(temp3);
	}
}
