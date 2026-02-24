OPENMP FFTW:
   so fftw uses the environment variable openmp nthreads for creating the plan and there is basically nothing you can do about it except setting that environment variable within the code. Either just let it run as is and use all threads or set environment variable

   setenv("OMP_NUM_THREADS", std::to_string(3).c_str(), 1);   //TESTVALUE  // Force OpenMP to 3 threads

VECTORIZATION:
   having built fftw with vectorization (dont know if it always builds with vectorization) it tests in the plan creation with different vectorization levels at runtime and then decides which is fastest for the given input sizes. Apparently atleas avx512 even thought its supported on this machine its not fast enough so during fft execution its actually not used


operations between cuda memcopy are cuda and cpu read/write operations
cube sizes must be power of 2 so that fft is fast. if e.g. size is prime then fft is 10x slower
CUDA:
   not actually worth to investigate any speedup or graph implementation, there is no overhead problem, its just the cufft kernel that takes very long


concerning the parallelization:


A: currently the image is cut into subimages and each of these runs on one thread.
B: One could also use omp / fftw multithreading to sequentially deconvolve each subimage and use all threads for that subimage, as many operations in the deconvolution allow concurrency.

Tradeoffs:
    - Both very similar in how long they need for the deconvolution
    - On cuda there is no difference as neither runs in parallel on cpu.
    - A seperates at a higher level but needs to cut images into subimages to achieve parallelization. B could possibly take in the entire image, if the memory allows it, and multithread that. 
       -> Due to padding A therefore has more individual computations as the size of (nthreads x subimagesize > originalImageSize) as each subimage needs its own pading which is the same shape as the padding of the originalimage. So in total there is more padding on the subimages than there is on the original image
    - B has a longer fftw plan initialization as the plan is much larger( as the subimage is much larger)



using testalgorithm, so no deconvolution grid and no grid are the same
when using a real deconvolutionalgorithm the results using grid and not grid are not the same
-> probably some normalization difference, e.g. some normalization is different because grid/data inside deconvolution is different

difference between using cuda and cpu is very minimal, prob some difference in double/floating point representation but in the order 1e-8
