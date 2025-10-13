concerning the parallelization:


A: currently the image is cut into subimages and each of these runs on one thread.
B: One could also use omp / fftw multithreading to sequentially deconvolve each subimage and use all threads for that subimage, as many operations in the deconvolution allow concurrency.

Tradeoffs:
    - Both very similar in how long they need for the deconvolution
    - On cuda there is no difference as neither runs in parallel on cpu.
    - A seperates at a higher level but needs to cut images into subimages to achieve parallelization. B could possibly take in the entire image, if the memory allows it, and multithread that. 
       -> Due to padding A therefore has more individual computations as the size of (nthreads x subimagesize > originalImageSize) as each subimage needs its own pading which is the same shape as the padding of the originalimage. So in total there is more padding on the subimages than there is on the original image
    - B has a longer fftw plan initialization as the plan is much larger( as the subimage is much larger)