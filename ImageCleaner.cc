#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <algorithm>
#include <cstring>
using namespace std;

const float PI = acos(-1.0);

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

void fft(float *real_image, float *imag_image, int n, int step, int isign = 1) {
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
  unsigned int eight = size_y / 8;
  unsigned int eight7 = size_y - eight;
  unsigned int filtered_size = eight7 - eight;

  #pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,size)
  for (x = 0; x < size; x++) {
    fft(real_image + x * size, imag_image + x * size, size, 1);
    memset(real_image + x * size + eight, 0, sizeof(float) * filtered_size);
    memset(imag_image + x * size + eight, 0, sizeof(float) * filtered_size);   
  }
  #pragma omp parallel for schedule(dynamic,chunk) private(x, y) shared(real_image,imag_image,size)
  for (y = 0; y < eight * 2; y++) {
    int iy = y < eight ? y : y + eight7 - eight;
    fft(real_image + iy, imag_image + iy, size, size);
    //Clear
    float *real_image_ptr  = real_image + eight * size + iy;
    float *imag_image_ptr = imag_image + eight * size + iy;
    for (x = 0; x < filtered_size; x++, real_image_ptr += size, imag_image_ptr += size) {
      *real_image_ptr = 0, *imag_image_ptr = 0;
    }
  }

  #pragma omp parallel for schedule(dynamic,chunk) private(y) shared(real_image,imag_image,size)
  for(y = 0; y < eight * 2; y++) {
    int iy = y < eight ? y : y + eight7 - eight;
    fft(real_image + iy, imag_image + iy, size, size, -1);
  }

  #pragma omp parallel for schedule(dynamic,chunk) private(x, y) shared(real_image,imag_image,size)
  for(x = 0; x < size; x++) {
    float factor = size * size;
    fft(real_image + x * size, imag_image + x * size, size, 1, -1);
    float *real_image_ptr  = real_image + x * size;
    float *imag_image_ptr = imag_image + x * size;
    for (y = 0; y < size; y++, real_image_ptr++, imag_image_ptr++) {
      *real_image_ptr /= factor, *imag_image_ptr /= factor;
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
