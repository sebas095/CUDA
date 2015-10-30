#include <bits/stdc++.h>
//#include <cuda.h>
#define H 10
#define W 10

using namespace std;

void foo(int* v){
  for(int i=0; i<H; i++){
    for(int j=0; j<W; j++){
      v[i*W+j] = 1;
    }
  }
}

void add(int *A, int *B,int *C){
  for(int i=0; i<H; i++){
    for(int j=0; j<W; j++){
      C[i*W+j] = A[i*W+j] + B[i*W+j];
    }
  }
}

void mostrar(int *v){
  for(int i=0; i<H; i++){
    for(int j=0; j<W; j++){
      cout<<v[i*W+j]<<" ";
    }
    cout<<endl;
  }
}

__global__ void sumMat(int *d_A, int *d_B,int *d_C){
  int i = blockIdx.y*blockDim.y+threadIdx.y;
  int j = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < H && j < W)
    d_C[i*W+j] = d_A[i*W+j] + d_B[i*W+j];
}


int main(){
  clock_t start, end;
  double cpu_time_used;
  int *A = (int*)malloc(H*W*sizeof(int));
  int *B = (int*)malloc(H*W*sizeof(int));
  int *C = (int*)malloc(H*W*sizeof(int));
  int *D = (int*)malloc(H*W*sizeof(int));

  foo(A);
  foo(B);
  start = clock();
  add(A,B,C);
  //mostrar(C);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido CPU = %lf s\n",cpu_time_used);

  int *d_A,*d_B,*d_D;
  float blockSize = 32;
  dim3 dimBlock(blockSize,blockSize);
  dim3 dimGrid(ceil(W/float(blockSize)),ceil(H/float(blockSize)),1);

  cudaMalloc((void**)&d_A,sizeof(int)*H*W);
  cudaMalloc((void**)&d_B,sizeof(int)*H*W);
  cudaMalloc((void**)&d_D,sizeof(int)*H*W);

  cudaMemcpy(d_A,A,sizeof(int)*H*W,cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,B,sizeof(int)*H*W,cudaMemcpyHostToDevice);

  start = clock();
  sumMat<<<dimGrid,dimBlock>>>(d_A,d_B,d_D);
  cudaMemcpy(D,d_D,sizeof(int)*H*W,cudaMemcpyDeviceToHost);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo invertido GPU = %lf s\n",cpu_time_used);

  //mostrar(D);
  free(A);
  free(B);
  free(C);
  free(D);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_D);

}