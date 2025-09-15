- fix psf preprocessing (e.g. creating the psfmap = old cubenumvec and implement layermap AND cubemap, currently not implemented in deconvolutionprocessor)

- test refactor with rldeconv

- refactor all other deconv algorithms

- rename gpu backend to cuda






a event system which lets lower layer windows be able to ocmmunicate with higher level would be nice







cubenumvec and layernumvec: what do they do
 LK these should allow differnt cubes and layers to be deconved with different psfs

secondarypsfcube / vec what do they do
 LK i think these are remains, replaced by cubenumvec to allow arbitrary number of psfs





