# CUDA-Based Image Processing

This project implements a set of image processing operations using **CUDA** to accelerate computations on the GPU.

## Features

- **Grayscale Conversion**
- **Gaussian Blur Filtering**
- **Laplacian Filtering**
- **Sobel Edge Detection**

## Optimizations

- Utilizes **constant memory** for filter kernels
- Leverages **shared memory** to improve convolution performance

## Technologies Used

- **CUDA**
- **C++**
- **OpenCV**
