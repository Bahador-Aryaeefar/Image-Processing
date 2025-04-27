#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include<iostream>
#include <stdio.h>
#include<cmath>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace std::chrono;
using namespace cv;

__global__ void colorToGray(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;

	if (Col < width && Row < height) {
		int greyOffset = Row * width + Col;
		int rgbOffset = greyOffset * 3;
		unsigned char r = Pin[rgbOffset + 0];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char b = Pin[rgbOffset + 2];
		Pout[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}
}

__global__ void simpleGuassConvo(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	float filter[5][5] = { {2,4,5,4,2}, {4,9,12,9,4},{5,12,15,12,5},{4,9,12,9,4},{2,4,5,4,2} };
	for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) {
		filter[i][j] = filter[i][j] / 159;
	}
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int n = 2;

	float Pvalue = 0;
	int Pnumber = 0;
	for (int mi = -n; mi <= n; mi++) {
		for (int mj = -n; mj <= n; mj++) {
			if (row + mi >= 0 && row + mi < width && col + mj >= 0 && col + mj < height) {
				Pvalue += Pin[((row + mi) * width + (col + mj))] * filter[mi + n][mj + n];
				Pnumber++;
			}
		}
	}
	Pout[(row * width + col)] = Pvalue;
}

__global__ void simpleLaplus(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	float filter[9][9] = {
			{0,0,3,2,2,2,3,0,0},
			{0,2,3,5,5,5,3,2},
			{3,3,5,3,0,3,5,3,3},
			{2,5,3,-12,-23,-12,3,5,2},
			{2,5,3,-12,-23,-12,-3,-5,-2},
			{3,3,5,3,0,3,5,3,3},
			{0,2,3,5,5,5,3,2,0},
			{0,0,3,2,2,3,0,0} };
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int n = 4;

	float Pvalue = 0;
	int Pnumber = 0;
	for (int mi = -n; mi <= n; mi++) {
		for (int mj = -n; mj <= n; mj++) {
			if (row + mi >= 0 && row + mi < width && col + mj >= 0 && col + mj < height) {
				Pvalue += Pin[((row + mi) * width + (col + mj))] * filter[mi + n][mj + n];
				Pnumber++;
			}
		}
	}
	Pout[(row * width + col)] = Pvalue / Pnumber;
}

__global__ void simpleSobelConvo(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	float filterx[3][3] = { {-0.25, 0, 0.25}, {-0.5, 0, 0.5}, {-0.5, 0, 0.5} };
	float filtery[3][3] = { {-0.25, -0.5, -0.25}, {0, 0, 0}, {0.25, 0.5, 0.25} };
	//float filterx[3][3] = { {-1,0,1},{2,0,2},{-1,0,1} };
	//float filtery[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	float Pvaluex = 0;
	float Pvaluey = 0;
	int Pnumber = 0;
	for (int mi = -1; mi <= 1; mi++) {
		for (int mj = -1; mj <= 1; mj++) {
			if (row + mi >= 0 && row + mi < width && col + mj >= 0 && col + mj < height) {
				Pvaluex += Pin[((row + mi) * width + (col + mj))] * filterx[mi + 1][mj + 1];
				Pvaluey += Pin[((row + mi) * width + (col + mj))] * filtery[mi + 1][mj + 1];
				Pnumber++;
			}
		}
	}
	Pout[(row * width + col)] = sqrt(Pvaluex * Pvaluex + Pvaluey * Pvaluey);
}

float laplus[9][9] = {
			{0,0,3,2,2,2,3,0,0},
			{0,2,3,5,5,5,3,2},
			{3,3,5,3,0,3,5,3,3},
			{2,5,3,-12,-23,-12,3,5,2},
			{2,5,3,-12,-23,-12,-3,-5,-2},
			{3,3,5,3,0,3,5,3,3},
			{0,2,3,5,5,5,3,2,0},
			{0,0,3,2,2,3,0,0} };
float guass[5][5] = { {2,4,5,4,2}, {4,9,12,9,4},{5,12,15,12,5},{4,9,12,9,4},{2,4,5,4,2} };
float sobelx[3][3] = { {-1,0,1},{2,0,2},{-1,0,1} };
float sobely[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
__constant__ float d_laplus[9][9];
__constant__ float d_guass[5][5];
__constant__ float d_sobelx[3][3];
__constant__ float d_sobely[3][3];

__global__ void constGuassConvo(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int n = 2;

	float Pvalue = 0;
	int Pnumber = 0;
	for (int mi = -n; mi <= n; mi++) {
		for (int mj = -n; mj <= n; mj++) {
			if (row + mi >= 0 && row + mi < width && col + mj >= 0 && col + mj < height) {
				Pvalue += Pin[((row + mi) * width + (col + mj))] * d_guass[mi + n][mj + n] / 159;
				Pnumber++;
			}
		}
	}
	Pout[(row * width + col)] = Pvalue;
}

__global__ void constSobelConvo(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	float Pvaluex = 0;
	float Pvaluey = 0;
	int Pnumber = 0;
	for (int mi = -1; mi <= 1; mi++) {
		for (int mj = -1; mj <= 1; mj++) {
			if (row + mi >= 0 && row + mi < width && col + mj >= 0 && col + mj < height) {
				Pvaluex += Pin[((row + mi) * width + (col + mj))] * d_sobelx[mi + 1][mj + 1];
				Pvaluey += Pin[((row + mi) * width + (col + mj))] * d_sobely[mi + 1][mj + 1];
				Pnumber++;
			}
		}
	}
	Pout[(row * width + col)] = sqrt(Pvaluex * Pvaluex + Pvaluey * Pvaluey);
}

__global__ void sharedGuassConvo(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = 2;

	__shared__ float shared[16][16];

	shared[i][j] = Pin[row * width + col];

	__syncthreads();

	float Pvalue = 0;
	int Pnumber = 0;
	for (int mi = -n; mi <= n; mi++) {
		for (int mj = -n; mj <= n; mj++) {
			if (row + mi >= 0 && row + mi < width && col + mj >= 0 && col + mj < height) {
				if (i + mi >= 0 && i + mi < 16 && j + mj >= 0 && j + mj < 16) {
					Pvalue += shared[i + mi][j + mj] * d_guass[mi + n][mj + n] / 159;
				}
				else Pvalue += Pin[((row + mi) * width + (col + mj))] * d_guass[mi + n][mj + n] / 159;
				Pnumber++;
			}
		}
	}
	Pout[(row * width + col)] = Pvalue;
}

__global__ void sharedSobelConvo(unsigned char* Pin, unsigned char* Pout, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	float Pvaluex = 0;
	float Pvaluey = 0;
	int Pnumber = 0;

	int i = threadIdx.x;
	int j = threadIdx.y;

	__shared__ float shared[16][16];

	shared[i][j] = Pin[row * width + col];

	__syncthreads();

	for (int mi = -1; mi <= 1; mi++) {
		for (int mj = -1; mj <= 1; mj++) {
			if (row + mi >= 0 && row + mi < width && col + mj >= 0 && col + mj < height) {
				if (i + mi >= 0 && i + mi < 16 && j + mj >= 0 && j + mj < 16) {
					Pvaluex += shared[i + mi][j + mj] * d_sobelx[mi + 1][mj + 1];
					Pvaluey += shared[i + mi][j + mj] * d_sobely[mi + 1][mj + 1];
				}
				else {
					Pvaluex += Pin[((row + mi) * width + (col + mj))] * d_sobelx[mi + 1][mj + 1];
					Pvaluey += Pin[((row + mi) * width + (col + mj))] * d_sobely[mi + 1][mj + 1];
				}
				Pnumber++;
			}
		}
	}
	Pout[(row * width + col)] = sqrt(Pvaluex * Pvaluex + Pvaluey * Pvaluey);
}

int main(void) {
	//serial code
	if (false) {
		string fileName = "lenna.tiff";
		Mat Pin = imread(fileName);
		imshow("imgae", Pin);
		printf("\n\n\Read file --------------------------------------\n\n");
		waitKey(0);
		int width = Pin.cols;
		int height = Pin.rows;
		int colorsize = width * height * 3 * sizeof(unsigned char);
		int graysize = width * height * sizeof(unsigned char);
		Mat PGray(width, height, CV_8UC1);
		unsigned char* d_Pin, * d_Pout;

		cudaMalloc((void**)&d_Pin, colorsize);
		cudaMalloc((void**)&d_Pout, graysize);
		cudaMemcpy(d_Pin, Pin.data, colorsize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Pout, PGray.data, graysize, cudaMemcpyHostToDevice);
		dim3 dimBlock(16, 16, 1);
		dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);
		colorToGray << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);
		cudaMemcpy(PGray.data, d_Pout, graysize, cudaMemcpyDeviceToHost);
		imshow("gray image", PGray);
		waitKey(0);


		Mat Pconvo(width, height, CV_8UC1);
		float filter1[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };
		float filter2[5][5] = { {2,4,5,4,2}, {4,9,12,9,4},{5,12,15,12,5},{4,9,12,9,4},{2,4,5,4,2} };
		float filter3[9][9] = {
			{0,0,3,2,2,2,3,0,0},
			{0,2,3,5,5,5,3,2},
			{3,3,5,3,0,3,5,3,3},
			{2,5,3,-12,-23,-12,3,5,2},
			{2,5,3,-12,-23,-12,-3,-5,-2},
			{3,3,5,3,0,3,5,3,3},
			{0,2,3,5,5,5,3,2,0},
			{0,0,3,2,2,3,0,0} };
		int n = 3;
		for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) {
			filter2[i][j] = filter2[i][j] / 159;
		}

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				float Pvalue1 = 0;
				float Pvalue2 = 0;
				float Pvalue3 = 0;
				int Pnumber = 0;
				for (int mi = -n; mi <= n; mi++) {
					for (int mj = -n; mj <= n; mj++) {
						if (i + mi >= 0 && i + mi < width && j + mj >= 0 && j + mj < height) {
							Pvalue1 += PGray.data[((i + mi) * width + (j + mj))] * filter3[mi + n][mj + n];
							Pnumber++;
						}
					}
				}
				Pconvo.data[(i * width + j)] = Pvalue1 / Pnumber;
			}
		}


		imshow("convo image", Pconvo);
		waitKey(0);


		printf("Convolotion on host --------------------------------------\n\n");
		Mat Pedge(width, height, CV_8UC1);
		float filterx[3][3] = { {-0.25, 0, 0.25}, {-0.5, 0, 0.5}, {-0.5, 0, 0.5} };
		float filtery[3][3] = { {-0.25, -0.5, -0.25}, {0, 0, 0}, {0.25, 0.5, 0.25} };
		//float filterx[3][3] = { {-1,0,1},{2,0,2},{-1,0,1} };
		//float filtery[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				float Pvaluex = 0;
				float Pvaluey = 0;
				int Pnumber = 0;
				for (int mi = -1; mi <= 1; mi++) {
					for (int mj = -1; mj <= 1; mj++) {
						if (i + mi >= 0 && i + mi < width && j + mj >= 0 && j + mj < height) {
							Pvaluex += Pconvo.data[((i + mi) * width + (j + mj))] * filterx[mi + 1][mj + 1];
							Pvaluey += Pconvo.data[((i + mi) * width + (j + mj))] * filtery[mi + 1][mj + 1];
							Pnumber++;
						}
					}
				}
				Pedge.data[(i * width + j)] = sqrt(Pvaluex * Pvaluex + Pvaluey * Pvaluey);
			}
		}

		imshow("edge image", Pedge);
		waitKey(0);
	}

	//simple krenel
	if (false) {
		string fileName = "lenna.tiff";
		Mat Pin = imread(fileName);
		imshow("imgae", Pin);
		printf("\n\n\Read file --------------------------------------\n\n");
		waitKey(0);
		int width = Pin.cols;
		int height = Pin.rows;
		int colorsize = width * height * 3 * sizeof(unsigned char);
		int graysize = width * height * sizeof(unsigned char);
		Mat PGray(width, height, CV_8UC1);

		{
			unsigned char* d_Pin, * d_Pout;
			cudaMalloc((void**)&d_Pin, colorsize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, Pin.data, colorsize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, PGray.data, graysize, cudaMemcpyHostToDevice);
			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);
			colorToGray << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);
			cudaMemcpy(PGray.data, d_Pout, graysize, cudaMemcpyDeviceToHost);
			imshow("gray image", PGray);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
		}

		Mat Pconvo(width, height, CV_8UC1);
		{
			int n = 3;
			unsigned char* d_Pin, * d_Pout;

			cudaMalloc((void**)&d_Pin, graysize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, PGray.data, graysize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, Pconvo.data, graysize, cudaMemcpyHostToDevice);

			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);

			//simpleGuassConvo << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);
			simpleLaplus << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);

			cudaMemcpy(Pconvo.data, d_Pout, graysize, cudaMemcpyDeviceToHost);

			imshow("convo image", Pconvo);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
		}

		Mat Pedge(width, height, CV_8UC1);
		{
			unsigned char* d_Pin, * d_Pout;
			cudaMalloc((void**)&d_Pin, graysize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, Pconvo.data, graysize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, Pedge.data, graysize, cudaMemcpyHostToDevice);

			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);

			simpleSobelConvo << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);

			cudaMemcpy(Pedge.data, d_Pout, graysize, cudaMemcpyDeviceToHost);


			imshow("edge image", Pedge);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
		}

	}

	//constant krenel 
	if (false) {
		string fileName = "lenna.tiff";
		Mat Pin = imread(fileName);
		imshow("imgae", Pin);
		printf("\n\n\Read file --------------------------------------\n\n");
		waitKey(0);
		int width = Pin.cols;
		int height = Pin.rows;
		int colorsize = width * height * 3 * sizeof(unsigned char);
		int graysize = width * height * sizeof(unsigned char);
		Mat PGray(width, height, CV_8UC1);

		{
			unsigned char* d_Pin, * d_Pout;
			cudaMalloc((void**)&d_Pin, colorsize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, Pin.data, colorsize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, PGray.data, graysize, cudaMemcpyHostToDevice);
			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);
			colorToGray << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);
			cudaMemcpy(PGray.data, d_Pout, graysize, cudaMemcpyDeviceToHost);
			imshow("gray image", PGray);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
		}

		Mat Pconvo(width, height, CV_8UC1);
		{
			int n = 3;
			unsigned char* d_Pin, * d_Pout;

			cudaMalloc((void**)&d_Pin, graysize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, PGray.data, graysize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, Pconvo.data, graysize, cudaMemcpyHostToDevice);

			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);

			cudaMemcpyToSymbol(d_guass, guass, sizeof(guass));

			constGuassConvo << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);

			cudaMemcpy(Pconvo.data, d_Pout, graysize, cudaMemcpyDeviceToHost);

			imshow("convo image", Pconvo);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
			cudaFree(d_guass);
		}

		Mat Pedge(width, height, CV_8UC1);
		{
			unsigned char* d_Pin, * d_Pout;
			cudaMalloc((void**)&d_Pin, graysize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, Pconvo.data, graysize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, Pedge.data, graysize, cudaMemcpyHostToDevice);

			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);

			cudaMemcpyToSymbol(d_sobelx, sobelx, sizeof(sobelx));
			cudaMemcpyToSymbol(d_sobely, sobely, sizeof(sobely));

			constSobelConvo << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);

			cudaMemcpy(Pedge.data, d_Pout, graysize, cudaMemcpyDeviceToHost);

			imshow("edge image", Pedge);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
			cudaFree(d_sobelx);
			cudaFree(d_sobely);
		}

	}

	//shared krenel
	if (false) {
		string fileName = "lenna.tiff";
		Mat Pin = imread(fileName);
		imshow("imgae", Pin);
		printf("\n\n\Read file --------------------------------------\n\n");
		waitKey(0);
		int width = Pin.cols;
		int height = Pin.rows;
		int colorsize = width * height * 3 * sizeof(unsigned char);
		int graysize = width * height * sizeof(unsigned char);
		Mat PGray(width, height, CV_8UC1);

		{
			unsigned char* d_Pin, * d_Pout;
			cudaMalloc((void**)&d_Pin, colorsize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, Pin.data, colorsize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, PGray.data, graysize, cudaMemcpyHostToDevice);
			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);
			colorToGray << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);
			cudaMemcpy(PGray.data, d_Pout, graysize, cudaMemcpyDeviceToHost);
			imshow("gray image", PGray);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
		}

		Mat Pconvo(width, height, CV_8UC1);
		{
			int n = 3;
			unsigned char* d_Pin, * d_Pout;

			cudaMalloc((void**)&d_Pin, graysize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, PGray.data, graysize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, Pconvo.data, graysize, cudaMemcpyHostToDevice);

			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);

			cudaMemcpyToSymbol(d_guass, guass, sizeof(guass));

			sharedGuassConvo << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);

			cudaMemcpy(Pconvo.data, d_Pout, graysize, cudaMemcpyDeviceToHost);

			imshow("convo image", Pconvo);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
			cudaFree(d_guass);
		}

		Mat Pedge(width, height, CV_8UC1);
		{
			unsigned char* d_Pin, * d_Pout;
			cudaMalloc((void**)&d_Pin, graysize);
			cudaMalloc((void**)&d_Pout, graysize);
			cudaMemcpy(d_Pin, Pconvo.data, graysize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_Pout, Pedge.data, graysize, cudaMemcpyHostToDevice);

			dim3 dimBlock(16, 16, 1);
			dim3 dimGrid(ceil((double)width / (double)dimBlock.x), ceil((double)height / (double)dimBlock.y), 1);

			cudaMemcpyToSymbol(d_sobelx, sobelx, sizeof(sobelx));
			cudaMemcpyToSymbol(d_sobely, sobely, sizeof(sobely));

			sharedSobelConvo << <dimGrid, dimBlock >> > (d_Pin, d_Pout, width, height);

			cudaMemcpy(Pedge.data, d_Pout, graysize, cudaMemcpyDeviceToHost);

			imshow("edge image", Pedge);
			waitKey(0);

			cudaFree(d_Pin);
			cudaFree(d_Pout);
			cudaFree(d_sobelx);
			cudaFree(d_sobely);
		}

	}
}