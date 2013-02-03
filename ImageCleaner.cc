#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>

#define PI	3.14159265

void cpu_fftx(float *real_image, float *imag_image, float *tmp_real_image, float *tmp_imag_image, int size_x, int size_y,float *cosIndex, float *sinIndex)
{
  // Create some space for storing temporary values
	int eightX = size_x/8;
	int eight7X = size_x - eightX;
	int eightY = size_y/8;
	int eight7Y = size_y - eightY;

  unsigned int x = 0;
	unsigned int chunk = size_x/4;
  #pragma omp parallel for schedule(dynamic,chunk) private(x) shared(real_image,imag_image,cosIndex,sinIndex,size_x,size_y)
  for(x = 0; x < size_x; x++)
  {
	int row=size_x*x;
	  
	float *realOutBuffer = new float[size_x];
	float *imagOutBuffer = new float[size_x]; 
	for(unsigned int y = 0; y < size_y; y++)
	{
	  // Compute the value for this index
	  realOutBuffer[y] = 0.0f;
	  imagOutBuffer[y] = 0.0f;
	  if (y<eightY || y>=eight7Y) {
		  for(unsigned int n = 0; n < size_y; n++)
		  {
			  realOutBuffer[y] += (real_image[row + n] * cosIndex[n+y*size_y]) - (imag_image[row + n] * sinIndex[n+y*size_y]);
			  imagOutBuffer[y] += (imag_image[row + n] * cosIndex[n+y*size_y]) + (real_image[row + n] * sinIndex[n+y*size_y]);
		  }
	  }
	}
	// Write the buffer back to were the original values were
	for(unsigned int t = 0; t < size_y; t++)
	{
	  tmp_real_image[t*size_x + x] = realOutBuffer[t];
	  tmp_imag_image[t*size_x + x] = imagOutBuffer[t];
	}
	delete [] realOutBuffer;
	delete [] imagOutBuffer;
  }
  // Reclaim some memory
}

// This is the same as the thing above, except it has a scaling factor added to it
void cpu_ifftx(float *real_image, float *imag_image, float *tmp_real_image, float *tmp_imag_image, int size_x, int size_y,float *cosIndex, float *sinIndex)
{
	int eightX = size_x/8;
	int eight7X = size_x - eightX;
	int eightY = size_y/8;
	int eight7Y = size_y - eightY;	
  // Create some space for storing temporary values
  unsigned int x = 0;
	unsigned int chunk = size_x/4;
  #pragma omp parallel for schedule(dynamic,4) private(x) shared(real_image,imag_image,cosIndex,sinIndex,size_x,size_y)
  for(x = 0; x < size_x; x++)
  {
	 
		float *realOutBuffer = new float[size_x];
		float *imagOutBuffer = new float[size_x];
	    int row = size_x*x;
		for(unsigned int y = 0; y < size_y; y++)
		{
		  // Compute the value for this index
		  realOutBuffer[y] = 0.0f;
		  imagOutBuffer[y] = 0.0f;
		  for(unsigned int n = 0; n < size_y; n++)
		  {
		  	if (n >= eightX  && n < eight7X)
		  		continue;
		  	else{		  			
			  realOutBuffer[y] += (real_image[row + n] * cosIndex[n+y*size_y]) - (imag_image[row + n] * sinIndex[n+y*size_y]);
			  imagOutBuffer[y] += (imag_image[row + n] * cosIndex[n+y*size_y]) + (real_image[row + n] * sinIndex[n+y*size_y]);
			  }
		  }

		  // Incoporate the scaling factor here
		  realOutBuffer[y] /= size_y;
		  imagOutBuffer[y] /= size_y;
		}
		// Write the buffer back to were the original values were
		for(unsigned int y = 0; y < size_y; y++)
		{
		  tmp_real_image[x + size_x*y] = realOutBuffer[y];
		  tmp_imag_image[x + size_x*y] = imagOutBuffer[y];
		}
		delete [] realOutBuffer;
		delete [] imagOutBuffer;
	
  }
  // Reclaim some memory
}

void cpu_ffty(float *real_image, float *imag_image, int size_x, int size_y,float *cosIndex, float *sinIndex)
{
  // Allocate some space for temporary values
	unsigned int y = 0;
	int eightX = size_x/8;
	int eight7X = size_x - eightX;
	int eightY = size_y/8;
	int eight7Y = size_y - eightY;
	unsigned int chunk = size_x/4;
  #pragma omp parallel for schedule(dynamic,chunk) private(y) shared(real_image,imag_image,cosIndex,sinIndex,size_x,size_y)
  
  for(y = 0; y < size_y; y++)
  {
	if (y<eightY || y>=eight7Y) {
		float *realOutBuffer = new float[size_y];
		float *imagOutBuffer = new float[size_y];
		
		for(unsigned int x = 0; x < size_x; x++)
		{
			realOutBuffer[x] = 0.0f;
			imagOutBuffer[x] = 0.0f;
			
			if(x < eightX || x >= eight7X)
			{
				// Compute the value for this index
				
				for(unsigned int n = 0; n < size_x; n++)
				{
					realOutBuffer[x] += (real_image[n*size_x + y] * cosIndex[n*size_x+x]) - (imag_image[n*size_x + y] * sinIndex[n*size_x+x]);
					imagOutBuffer[x] += (imag_image[n*size_x + y] * cosIndex[n*size_x+x]) + (real_image[n*size_x + y] * sinIndex[n*size_x+x]);
				}
			}	
		}
		// Write the buffer back to were the original values were
		for(unsigned int x = 0; x < size_x; x++)
		{
			real_image[x*size_x + y] = realOutBuffer[x];
			imag_image[x*size_x + y] = imagOutBuffer[x];
		}
		delete [] realOutBuffer;
		delete [] imagOutBuffer;
	}  

  }
  // Reclaim some memory
}

// This is the same as the thing about it, but it includes a scaling factor
void cpu_iffty(float *real_image, float *imag_image, int size_x, int size_y,float *cosIndex, float *sinIndex)
{
  // Create some space for storing temporary values
  unsigned int y = 0;
  unsigned int chunk = size_x/4;
  #pragma omp parallel for schedule(dynamic,chunk) private(y) shared(real_image,imag_image,cosIndex,sinIndex,size_x,size_y)
  for(y = 0; y < size_y; y++)
  {
	  float *realOutBuffer = new float[size_y];
	  float *imagOutBuffer = new float[size_y];
    for(unsigned int x = 0; x < size_x; x++)
    {

      // Compute the value for this index
      realOutBuffer[x] = 0.0f;
      imagOutBuffer[x] = 0.0f;
      for(unsigned int n = 0; n < size_x; n++)
      {
	realOutBuffer[x] += (real_image[n*size_x + y] * cosIndex[n*size_x+x]) - (imag_image[n*size_x + y] * sinIndex[n*size_x+x]);
	imagOutBuffer[x] += (imag_image[n*size_x + y] * cosIndex[n*size_x+x]) + (real_image[n*size_x + y] * sinIndex[n*size_x+x]);																										
      }

      // Incorporate the scaling factor here
      realOutBuffer[x] /= size_x;
      imagOutBuffer[x] /= size_x;
    }
    // Write the buffer back to were the original values were
    for(unsigned int x = 0; x < size_x; x++)
    {
      real_image[x*size_x + y] = realOutBuffer[x];
      imag_image[x*size_x + y] = imagOutBuffer[x];
    }
	  delete [] realOutBuffer;
	  delete [] imagOutBuffer;
  }
  // Reclaim some memory
}

void cpu_filter(float *real_image, float *imag_image, int size_x, int size_y)
{
  int eightX = size_x/8;
  int eight7X = size_x - eightX;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;
	unsigned int x = 0;
  #pragma omp parallel for private(x)
  for(x = 0; x < size_x; x++)
  {
    for(unsigned int y = 0; y < size_y; y++)
    {
      if(!(x < eightX && y < eightY) &&
	 !(x < eightX && y >= eight7Y) &&
	 !(x >= eight7X && y < eightY) &&
	 !(x >= eight7X && y >= eight7Y))
      {
		  // Zero out these values
		  real_image[y*size_x + x] = 0;
		  imag_image[y*size_x + x] = 0;
      }
    }
  }
}

void initCosSin(float *cosIndex1, float *sinIndex1,float *cosIndex2, float *sinIndex2, int K, int N)
{
	for(unsigned int n = 0; n < N; n++)
	{
		for(unsigned int k = 0; k < K; k++)
		{
			float term = -2 * PI * k * n / N;
			cosIndex1[n*N+k] = cos(term);
			sinIndex1[n*N+k] = sin(term);
			cosIndex2[n*N+k] = cos(-term);
			sinIndex2[n*N+k] = sin(-term);
		}
	}
}

float imageCleaner(float *real_image, float *imag_image, float *tmp_real_image, float *tmp_imag_image,int size_x, int size_y)
{
  // These are used for timing
  struct timeval tv1, tv2;
  struct timezone tz1, tz2;
	//Code Here
  float *cosIndex1 = new float[size_x*size_y];
  float *sinIndex1 = new float[size_x*size_y];
  float *cosIndex2 = new float[size_x*size_y];
  float *sinIndex2 = new float[size_x*size_y];
  initCosSin(cosIndex1, sinIndex1, cosIndex2, sinIndex2, size_x,size_y);

  // Start timing
  gettimeofday(&tv1,&tz1);

  // Perform fft with respect to the x direction
  cpu_fftx(real_image, imag_image, tmp_real_image, tmp_imag_image, size_x, size_y, cosIndex1, sinIndex1);
  cpu_fftx(tmp_real_image, tmp_imag_image, real_image, imag_image, size_x, size_y, cosIndex1, sinIndex1);
  // Perform fft with respect to the y direction
  // cpu_ffty(real_image, imag_image, size_x, size_y, cosIndex1, sinIndex1);

  // Filter the transformed image
  //cpu_filter(real_image, imag_image, size_x, size_y);

  // Perform an inverse fft with respect to the x direction
  cpu_ifftx(real_image, imag_image,tmp_real_image, tmp_imag_image, size_x, size_y, cosIndex2, sinIndex2);
  cpu_ifftx(tmp_real_image, tmp_imag_image,real_image, imag_image, size_x, size_y, cosIndex2, sinIndex2);
  // Perform an inverse fft with respect to the y direction
  //cpu_iffty(real_image, imag_image, size_x, size_y, cosIndex2, sinIndex2);

  // cpu_ifftx(real_image, imag_image, size_x, size_y, cosIndex2, sinIndex2);
  //cpu_iffty(real_image, imag_image, size_x, size_y, cosIndex2, sinIndex2);

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
