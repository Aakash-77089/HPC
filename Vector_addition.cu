#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void vectorAdd(int*a, int*b, int*c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 10;
    size_t size = n * sizeof(int);

    int *h_A, *h_B, *h_c;

    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_c = (int*)malloc(size);

    for(int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyHostToDevice);

    cout << "Result vector: \n";
    for(int i = 0; i < n; i++) {
        cout << h_c[i] << " ";
    }

    cout << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_A);
    free(h_B);
    free(h_c);

    return 0;
}