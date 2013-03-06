CS 149 Programming Assignment 4

Botao Hu (botaohu@stanford.edu)

[Optimization]

1. Implement 1D fast fourier transformation
Instead of using the original O(n^2) fourier transformation, we implement the fast fourier transformation of O(n log n), which significantly improves the performance. 

void fft(float *real_image, float *imag_image, int n, int step, int isign = 1) 

The input with an array [real_image, imag_image] of size [n] and the step in the array is [step].
[isign] represents the direction of fourier transformation. [isign] = -1 means the inverse fourier transformation. 

Reference:
Fast Fourier transform program, four1, from "Numerical Recipes in C" (Cambridge
Univ. Press) by W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery

2. Parallelized for by OpenMP
In order to compute 2D FFT, we first do row-wise transform and then do column-wise transform. 
For row-wise transform, we enumerate the row and input that row into 1D FFT. 
The calculation of rows are mutually independent and can be computed separately and parallelly.
Thus, we added OpenMP "dynamic parallelized for" operation at the loop of the enumeration of rows.
"dynamic" helps to balance the work load between threads. 
It's the similar case for column-wise transformation. 

3. Prune 3/4 y-dimension calculation due to low-pass filter
After calculating all 1-D FT of every rows, it is not necessary to compute FT of every columns, because low-pass filter will filter out the 3/4 of them. So we just compute FTs of the first 1/8 and last 1/8 columns. 

[Performance]

noisy_01:
Reference Kernel Execution Time: 96293.445312 ms
Optimized Kernel Execution Time: 41.956001 ms
Speedup: 2295.11

noisy_02:
Reference Kernel Execution Time: 96401.265625 ms
Optimized Kernel Execution Time: 42.112999 ms
Speedup: 2289.11

noisy_03:
Reference Kernel Execution Time: 785200.187500 ms
Optimized Kernel Execution Time: 202.889008 ms
Speedup: 3870.1

[Correctness]
The result of CpuReference and our method are visually identical.
The difference between the results are acceptable due to the error of the numerical calcluation. 

