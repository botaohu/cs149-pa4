#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>

#define PI	3.14159265

void fft(float *real_image, float *imag_image, int n, int step, int isign = 1) {
    int n, mmax, m, j, istep, i;
    float wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;
    
    for (i = 1, j = 1; i < n; i++) {
        if (j > i) {
            swap(real_image[j * step], real_image[i * step]);
            swap(imag_image[j * step], imag_image[i * step]);
        }
        int m;
        for (m = n >> 1; m >= 2 && j > m; j -= m, m >>= 1);
        j += m;
    }

    mmax = 1;
    while (n > mmax) {
      istep = mmax << 1;
      theta = isign * PI / mmax;
      wtemp = sin(0.5 * theta);
      wpr = -2.0 * wtemp * wtemp;
      wpi = sin(theta);
      wr = 1.0;
      wi = 0.0;
      for (m = 1; m < mmax; m++) {
          for (i = m; i <= n; i += istep) {
              j = i + mmax;
              tempr = wr * real_image[j * step] - wi * imag_image[j * step];
              tempi = wr * imag_image[j * step] + wi * real_image[j * step];
              real_image[j * step] = real_image[i * step] - tempr;
              imag_image[j * step] = imag_image[i * step] - tempi;
              real_image[i * step] += tempr;
              imag_image[i * step] += tempi;
          }
          wtemp = wr;
          wr = wtemp * wpr - wi * wpi + wr;
          wi = wi * wpr + wtemp * wpi + wi;
      }
      mmax = istep;
    }
}

float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y)
{
  // These are used for timing
  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  // Start timing
  gettimeofday(&tv1,&tz1);

  unsigned int chunk, x, y, size;

  size = size_x; 
  chunk = size / 8;

  #pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,cos_calc,sin_calc,size)
  for(x = 0; x < size; x++) {
    fft(real_image + x * size, imag_image + x * size, size, 1);
  }

  #pragma omp parallel for schedule(dynamic,chunk) private(y) shared(real_image,imag_image,cos_calc,sin_calc,size)
  for(y = 0; y < size_y; y++) {
    fft(real_image, imag_image, size, size);
  }
  
  //Clear
  unsigned int eight = size_y / 8;
  unsigned int eight7 = size_y - eight;
  unsigned int filtered_size = eight7 - eight;
  #pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,cos_calc,sin_calc,size,eight,eight7,filtered_size)
  for(x = 0; x < size; x++) {
    if (x < eight || x >= eight7) {
      memset(real_image + (x + eight) * size, 0, sizeof(float) * filtered_size);
      memset(imag_image + (x + eight) * size, 0, sizeof(float) * filtered_size);
    } else {
      memset(real_image + x * size, 0, sizeof(float) * size);
      memset(imag_image + x * size, 0, sizeof(float) * size);
    }
  }

  #pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,cos_calc,sin_calc,size)
  for(x = 0; x < size; x++) {
    fft(real_image + x * size, imag_image + x * size, size, 1, -1);
  }

  #pragma omp parallel for schedule(dynamic,chunk) private(y) shared(real_image,imag_image,cos_calc,sin_calc,size)
  for(y = 0; y < size_y; y++) {
    fft(real_image, imag_image, size, size, -1);
  }

  // End timing
  gettimeofday(&tv2,&tz2);

  // Compute the time difference in micro-seconds
  float execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel Execution Time: %f ms\n\n", execution);
  return execution;
}
