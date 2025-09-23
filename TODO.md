- use my own complex datatype for both backends and throughout
- maybe each deconvprocessor has a default cpu backend for preprocessing and then a second workerbackend
- who should set backend type, setupconfig or deconvconfig

- test refactor with rldeconv

- refactor all other deconv algorithms

- rename gpu backend to cuda

- in deconvconfig what is subimagesize and what is cubesize in original implementation




a event system which lets lower layer windows be able to ocmmunicate with higher level would be nice







cubenumvec and layernumvec: what do they do
 LK these should allow differnt cubes and layers to be deconved with different psfs

secondarypsfcube / vec what do they do
 LK i think these are remains, replaced by cubenumvec to allow arbitrary number of psfs





