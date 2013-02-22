#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <algorithm>
using namespace std;

const float PI = acos(-1.0);

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr


void fft(float *real_image, float *imag_image, int size, int step, int isign = 1)
{
  // Create some space for storing temporary values
  float *realOutBuffer = new float[size];
  float *imagOutBuffer = new float[size];
  // Local values
  float *fft_real = new float[size];
  float *fft_imag = new float[size];

   for(unsigned int y = 0; y < size; y++)
    {
      // Compute the frequencies for this index
      for(unsigned int n = 0; n < size; n++)
      {
        float term = -2 * PI * y * n / size * isign;
        fft_real[n] = cos(term);
        fft_imag[n] = sin(term);
      }

      // Compute the value for this index
      realOutBuffer[y] = 0.0f;
      imagOutBuffer[y] = 0.0f;
      for(unsigned int n = 0; n < size; n++)
      {
         realOutBuffer[y] += (real_image[n * step] * fft_real[n]) - (imag_image[n* step] * fft_imag[n]);
          imagOutBuffer[y] += (imag_image[n * step] * fft_real[n]) + (real_image[n * step] * fft_imag[n]);
      }
    }
    // Write the buffer back to were the original values were
    for(unsigned int y = 0; y < size; y++)
    {
      real_image[y * step] = realOutBuffer[y];
      imag_image[y * step] = imagOutBuffer[y];
    }
    // Reclaim some memory
  delete [] realOutBuffer;
  delete [] imagOutBuffer;
  delete [] fft_real;
  delete [] fft_imag;
}
void fft2(float *real_image, float *imag_image, int n, int step, int isign = 1) {
  real_image -= step;
  imag_image -= step;
  int mmax, m, j, istep, i, nstep, istep2;
  float wr, wi, tempr, tempi;

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
    nstep = n * step;
    istep = mmax * step;
    istep2 = istep * 2;
    for (m = 1; m <= mmax; m++) {
      wr = cos(PI * isign / mmax * m);
      wi = sin(PI * isign / mmax * m);
      for (i = m * step; i <= nstep; i += istep2) {
        j = i + istep;
        tempr = wr * real_image[j] - wi * imag_image[j];
        tempi = wr * imag_image[j] + wi * real_image[j];
        real_image[j] = real_image[i] - tempr;
        imag_image[j] = imag_image[i] - tempi;
        real_image[i] += tempr;
        imag_image[i] += tempi;
      }
    }
    mmax = mmax << 1;
  }
}


float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y)
{
  // These are used for timing
  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  // Start timing
  gettimeofday(&tv1,&tz1);

  int chunk, x, y, size;

  size = size_x; 
  chunk = size / 8;
  #pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,size)
  for (x = 0; x < size; x++) {
    fft2(real_image + x * size, imag_image + x * size, size, 1);
  }
  #pragma omp parallel for schedule(dynamic,chunk) private(y) shared(real_image,imag_image,size)
  for (y = 0; y < size; y++) {
    fft2(real_image + y, imag_image + y, size, size);
  }
  
  //Clear
  unsigned int eight = size_y / 8;
  unsigned int eight7 = size_y - eight;
  unsigned int filtered_size = eight7 - eight;
  #pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,size,eight,eight7,filtered_size)
  for (x = 0; x < size; x++) {
    if (x < eight || x >= eight7) {
      memset(real_image + x * size + eight, 0, sizeof(float) * filtered_size);
      memset(imag_image + x * size + eight, 0, sizeof(float) * filtered_size);
    } else {
      memset(real_image + x * size, 0, sizeof(float) * size);
      memset(imag_image + x * size, 0, sizeof(float) * size);
    }
  }
  /*
  //#pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,size,eight,eight7,filtered_size)
  for (x = 0; x < eight; x++) {
      memset(real_image + x * size + eight, 0, sizeof(float) * filtered_size);
      memset(imag_image + x * size + eight, 0, sizeof(float) * filtered_size);
  }
  for (x = eight7; x < size; x++) {
      memset(real_image + x * size + eight, 0, sizeof(float) * filtered_size);
      memset(imag_image + x * size + eight, 0, sizeof(float) * filtered_size);
  }

  memset(real_image + eight * size, 0, sizeof(float) * size * filtered_size);
      memset(imag_image + eight * size, 0, sizeof(float) * size * filtered_size);
    }
  }
  */


  #pragma omp parallel for schedule(dynamic,chunk) private(x, y) shared(real_image,imag_image,size)
  for(x = 0; x < size; x++) {
    fft2(real_image + x * size, imag_image + x * size, size, 1, -1);
    for (y = 0; y < size; y++) {
      *(real_image + x * size + y) /= size;
      *(imag_image + x * size + y) /= size;
    }
  }

  #pragma omp parallel for schedule(dynamic,chunk) private(x, y) shared(real_image,imag_image,size)
  for(y = 0; y < size; y++) {
    fft2(real_image + y, imag_image + y, size, size, -1);
    for (x = 0; x < size; x++) {
      *(real_image + y + x * size) /= size;
      *(imag_image + y + x * size) /= size;
    }
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
