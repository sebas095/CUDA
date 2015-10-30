#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

void foo(int* v, int size){
  for(int i=0; i<size; i++){
  	v[i] = 1;
  }
}

void add(int *v1, int *v2,int size, int *result){
	 for(int i=0; i<size; i++){
  	  result[i] = v1[i] + v2[i];
   }
}

void mostrar(int* v, int size){
  for(int i=0; i<size; i++){
  	cout<<v[i]<<endl;
  }
}

void compareTo(int *v, int *d_v, int size){
  bool flag = false;
  string msj="";
	 for(int i=0; i<size; i++){
  	  if(v[i] == d_v[i])continue;
     	else{
        flag=false;
     }
   }
  flag = true;
  msj = (flag)? "Son iguales!" : "Incoherencia!";
  cout<<msj<<endl;

}

__global__ void addCuda(int *d_v1, int *d_v2,int size, int *d_result){
	int i = threadIdx.x;
  if(i < size)d_result[i] = d_v1[i] + d_v2[i];
}

__global__ void addCuda2(int *d_v1, int *d_v2,int size, int *d_result){
	int i = blockIdx.x;
  if(i < size)d_result[i] = d_v1[i] + d_v2[i];
}

__global__ void addCuda3(int *d_v1, int *d_v2,int size, int *d_result){
	int i = threadIdx.x+(blockIdx.x*blockDim.x);
  if(i < size)d_result[i] = d_v1[i] + d_v2[i];
}

int main(){
  clock_t start, end;
	double cpu_time_used;
  int size_vec = 2000000000;
	int *vec1 = (int*)malloc(sizeof(int)*size_vec);
  int *vec2 = (int*)malloc(sizeof(int)*size_vec);
  int *vec3 = (int*)malloc(sizeof(int)*size_vec);
  int *vec4 = (int*)malloc(sizeof(int)*size_vec);

  foo(vec1,size_vec);
  foo(vec2,size_vec);
  start = clock();
  add(vec1,vec2,size_vec,vec3);
  //mostrar(vec3,size_vec);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Tiempo invertido CPU = %lf s\n",cpu_time_used);

  int *d_vec1,*d_vec2,*d_vec4;
  float blockSize = 1024;
  dim3 dimBlock(blockSize,1);
  dim3 dimGrid(ceil(size_vec/float(blockSize)),1,1);

  cudaMalloc((void**)&d_vec1,sizeof(int)*size_vec);
  cudaMalloc((void**)&d_vec2,sizeof(int)*size_vec);
  cudaMalloc((void**)&d_vec4,sizeof(int)*size_vec);

  cudaMemcpy(d_vec1,vec1,sizeof(int)*size_vec,cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec2,vec2,sizeof(int)*size_vec,cudaMemcpyHostToDevice);

  start = clock();
  //addCuda1<<<1,size_vec>>>(d_vec1,d_vec2,size_vec,d_vec4);
  //addCuda2<<<size_vec,1>>>(d_vec1,d_vec2,size_vec,d_vec4);
  addCuda3<<<dimGrid,dimBlock>>>(d_vec1,d_vec2,size_vec,d_vec4);
  cudaMemcpy(vec4,d_vec4,sizeof(int)*size_vec,cudaMemcpyDeviceToHost);
  end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Tiempo invertido GPU = %lf s\n",cpu_time_used);

  //mostrar(vec4,size_vec);
  //compareTo(vec3,vec4,size_vec);
  free(vec1);
  free(vec2);
  free(vec3);
  free(vec4);

  cudaFree(d_vec1);
  cudaFree(d_vec2);
  cudaFree(d_vec4);

}