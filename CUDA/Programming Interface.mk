### CUDA C Runtime
All its entry points are prefixed with **cuda**.
#### Initialization
It initializes the first time a runtime function is called.During initialization, the runtime creates a CUDA contex(»·¾³)for each device in the system.
When a host thread calls **cudaDeviceReset()**,this destroys the primary context of the device that the host thread currently operates on.
#### Device Memory
Device memory can be allocated either as linear memory or as CUDA arrays.
Linear memory is typically allocated using **cudaMalloc()** and freed using **cudaFree()** and data transfer between host memory and device memory are typically done using **cudaMemcpy()**.
```
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
	C[i] = A[i] + B[i];
}
// Host code
int main()
{
	int N = ...;
	size_t size = N * sizeof(float);
    // Allocate input vectors h_A and h_B in host memory
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	// Initialize input vectors
	...
	// Allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid =
		(N + threadsPerBlock - 1) / threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// Free host memory
	...
}
```
Linear memory can also be allocated through **cudaMallocPitch()** and **cudaMalloc3D()**.
#### Shared Memory
shared memory is allocated using the **__shared__** qualifier.And shared memory is expected to be much faster than global memory. Therefore, any opportunity to replace global memory accesses by shared memory accessed should be exploited.

#### page-locked Host Memory
```
cudaHostAlloc()
cudaFreeHost()
```
The benefits of using page-locked host memory.

### Asynchronous Concurrent Execution
CUDA exposes the following operations as independent tasks that can operate concurrently with one another
1.Computation on the host
2.Computation on the device
3.Memory transfers from the host to the device 
4.Memory transfers from the device to the hsot
5.Memory transfers within the memory of a given device
6.Memory transfers among devices
