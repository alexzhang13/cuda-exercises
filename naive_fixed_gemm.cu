#include <cuda_runtime.h>
#include <iostream>

__global__ void sgemm_naive (int M, int N, int K, float alpha, float* A, float* B, float beta, float *C) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < M && y < N) {
		float tmp = 0.0;
		for (int i = 0; i < K; i++) {
			// A is M x K. So we index across K. Maybe rename to row/col later.
			// B is K x N. So we index across K.
			tmp += A[x * K + i] * B[y + i * N];
		}
		// C is M x N
		C[y + x * N] = alpha * tmp + beta * C[y + x * N];
	}

}

int CEIL_DIV(int sz, int div) {
	return sz / div + (sz % div != 0);
}

void fill2D(float *A, int M, int N, int val) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			*(A + i + j * M) = val;
		}
	}
}

void print(float *A, int M, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			std::cout << A[i + j * N] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char* argv[]) {

	const int N = 4092;
	const int M = 4092;
	const int K = 4092;

	float *A, *B, *C;
	cudaMallocManaged((void **)&A, (N * K) * sizeof(int));
	cudaMallocManaged((void **)&B, (M * K) * sizeof(int));
	cudaMallocManaged((void **)&C, (M * N) * sizeof(int));

	cudaDeviceSynchronize();
	fill2D(A, M, K, 1.5f);
	fill2D(B, K, N, 5.9f);
	fill2D(C, M, N, 12.0f);

	dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
	dim3 blockDim(32, 32, 1); // act over 32 x 32 block of C

	float alpha = 0.5;
	float beta = 0.5;
	sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

	cudaDeviceSynchronize();

	// print(C, M, N);

	return 0;
}
