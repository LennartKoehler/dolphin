using testalgorithm, so no deconvolution grid and no grid are the same
when using a real deconvolutionalgorithm the results using grid and not grid are not the same
-> probably some normalization difference, e.g. some normalization is different because grid/data inside deconvolution is different

difference between using cuda and cpu is very minimal, prob some difference in double/floating point representation but in the order 1e-8
