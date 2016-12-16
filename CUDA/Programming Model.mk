### Kernels
    A kernel is defined by using the **__global** declaration specifier and the number of threads that execute that kernel for given kernel call is specified using a new **<<<...>>>** execution configuration syntax.
    As a illustration.
```
__global__ void vectorAdd(float *A, float *B, float *C, int N)
{
    int i = threadIdx.x;
    if(i < N)
	C[i] = A[i] + B[i];
}
int main()
{
...
    vector<<<1,N>>>(A,B,C,N);
...
}
```
**threadIdx** : Each thread that executes the kernel is given a unique thread ID;Each thread is accessible with the kernel through the **threadIdx**.
### Thread Hierarchy
The **threadIdx** is a 3-component vector.So threads can ben identified using one-dimensional,two-dimensional,three-dimensional thread index,forming one-dimensional,two-dimensional,or three-dimensional block of threads.
The index of a thread and its thread ID relate to each other:
for one-dimensional block, they are the same.
for two-dimensional block of size(Dx,Dy),the thread ID of index (x,y) is (x).
```
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
int i = threadIdx.x;
int j = threadIdx.y;
C[i][j] = A[i][j] + B[i][j];
}
int main()
{
...
// Kernel invocation with one block of N * N * 1 threads
int numBlocks = 1;
dim3 threadsPerBlock(N, N);
MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
...
}	
```
On current GPUs,a thread block may contain up to 1024 threads.
### the Grid of Thread Blocks
As the two examples above, we can learn that the number of threads per block and the number of blocks per grid specified in the **<<<...>>>** syntax can be of type int or dim3.
```
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
int i = threadIdx.x;
int j = threadIdx.y;
C[i][j] = A[i][j] + B[i][j];
}
int main()
{
...
// Kernel invocation with one block of N * N * 1 threads
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlocks);
MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
...
}
```
Thread Blocks are required to execute independently: It must possible to execute them in any order,in parallel or in series.
Threads within a block can cooperate by sharing data through some **shared memory** and by synchronizing their execution to coordinate memory accesses.Bying calling the **_syncthreads()** intrinsic function, one can specify synchronization points in the kernel.
### Memory Hierarchy
Each thread has private local memory.Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block.All threads have access to the same global memory.Also, there are two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces.
### Heterogeneous[“Ïππ]Programming
The CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C program.This is the case when the kernels execute on a GPU and the rest of the C program execute on a CPU.
The CUDA programming model assumes that both the host and the device maintain their own separete memory spaces in DRAM,referred to as **host memory** and **device memory**.Therefore, a program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime. This includes device memory allocation and deallocation as well as data transfer between host and device memory.

### Compute Capability
http://developer.nvidia.com/cuda-gpus
