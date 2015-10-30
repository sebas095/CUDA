#include <cstdio>
#include <ctime>
#include <cv.h>
#include <highgui.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;

__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if ((row < height) && (col < width)){
		imageOutput[row*width + col] = imageInput[(row*width + col) * 3 + RED] * 0.299 + imageInput[(row*width + col) * 3 + GREEN] * 0.587 \
			+ imageInput[(row*width + col) * 3 + BLUE] * 0.114;
	}
}

__global__ void sobelFilter(unsigned char *imageInput, int width, int height,
	unsigned int maskWidth, char *M, unsigned char *imageOutput) {
	unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

	int Pvalue = 0;

	int N_start_point_row = row - (maskWidth / 2);
	int N_start_point_col = col - (maskWidth / 2);

	for (int i = 0; i < maskWidth; i++){
		for (int j = 0; j < maskWidth; j++){
			if ((N_start_point_col + j >= 0 && N_start_point_col + j < width) \
				&& (N_start_point_row + i >= 0 && N_start_point_row + i < height)){
				Pvalue += imageInput[(N_start_point_row + i)*width + (N_start_point_col + j)] * M[i*maskWidth + j];
			}
		}
	}

	unsigned char ret;
	if (Pvalue < 0)
		Pvalue = 0;
	else if (Pvalue > 255)
		Pvalue = 255;
	ret = static_cast<unsigned char>(Pvalue);

	imageOutput[row*width + col] = ret;
}

void cudaCheckError(cudaError error, const char* error_msg) {
	if(error != cudaSuccess){
        printf("%s\n", error_msg);
        exit(-1);
    }
}

int main(int argc, char* argv[]) {
	cudaError_t error = cudaSuccess;
    clock_t start, end, startGPU, endGPU;
    double cpu_time_used, gpu_time_used;
    char h_Mask[] = {-1, 0, 1,
                     -2, 0, 2,
                     -1, 0, 1};
    char *d_Mask;
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput;
    Mat image = imread("./inputs/img1.jpg", CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Size s = image.size();

    int width = s.width;
    int height = s.height;
    dataRawImage = image.data;

    Mat gray_image;
    gray_image.create(height, width, CV_8UC1);
    h_imageOutput = gray_image.data;

    int size = sizeof(unsigned char) * width * height * image.channels();
    int sizeGray = sizeof(unsigned char) * width * height;

    error = cudaMalloc((void**)&d_dataRawImage, size);
    cudaCheckError(error, "Error reservando memoria para d_dataRawImage");

    error = cudaMalloc((void**)&d_imageOutput, sizeGray);
    cudaCheckError(error, "Error reservando memoria para d_imageOutput");

    error = cudaMalloc((void**)&d_Mask, sizeof(char) * 9);
    cudaCheckError(error, "Error reservando memoria para d_Mask");

    error = cudaMalloc((void**)&d_sobelOutput, sizeGray);
    cudaCheckError(error, "Error reservando memoria para d_sobelOutput");

	int blockSize = 32;
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);

    startGPU = clock();
    error = cudaMemcpy(d_dataRawImage, dataRawImage, size, cudaMemcpyHostToDevice);
    cudaCheckError(error, "Error copiando los datos de dataRawImage a d_dataRawImage ");

    error = cudaMemcpy(d_Mask, h_Mask, sizeof(char)*9, cudaMemcpyHostToDevice);
    cudaCheckError(error, "Error copiando los datos de h_Mask a d_Mask ");

    img2gray<<<dimGrid, dimBlock>>>(d_dataRawImage, width, height, d_imageOutput);
    cudaDeviceSynchronize();
    sobelFilter<<<dimGrid, dimBlock>>>(d_imageOutput, width, height, 3, d_Mask, d_sobelOutput);
    cudaDeviceSynchronize();
    cudaMemcpy(h_imageOutput, d_sobelOutput, sizeGray, cudaMemcpyDeviceToHost);
    endGPU = clock();

    start = clock();
    Mat gray_image_opencv, grad_x, abs_grad_x;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
    Sobel(gray_image_opencv, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    end = clock();

    imwrite("./outputs/Img1_cuda.png", gray_image);
    imwrite("./outputs/Img1_opencv.png", abs_grad_x);

    //namedWindow(img_file, WINDOW_NORMAL);
    //namedWindow("Gray Image CUDA", WINDOW_NORMAL);
    //namedWindow("Sobel Image OpenCV", WINDOW_NORMAL);

    //imshow(img_file, image);
    //imshow("Gray Image CUDA", gray_image);
    //imshow("Sobel Image OpenCV", abs_grad_x);

    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo Paralelo: %.10f\n", gpu_time_used);
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo OpenCV: %.10f\n", cpu_time_used);
    printf("La aceleracion obtenida es de %.10fX\n", cpu_time_used/gpu_time_used);

    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
    cudaFree(d_Mask);
    cudaFree(d_sobelOutput);

	//waitKey(0);

	return 0;
}