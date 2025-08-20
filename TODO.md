
make psf config loading and algorithm generation in dolphin more general and abstract
make reader/writer, might need that for gui


PSF:

create psfgenerator inside dolphin or seperately? do they share libraries/functionalities? perhaps make seperate like cube but include here

see psfgenerator by the people that created deconvolutionlab2, this can also be used for the development of the psf gen algorithms

what to think about:
- is the psf generation computationally expensive?
- gpu?
- input/output format
- how will it be used? standalone?



general PSF structure:

a psf generator always works together with its own psf config
where the psf config will hold the data it needs and can be created from a json and the psf generator only uses that for all the data management, but doesnt store data itself (except for the config)