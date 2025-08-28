
when in mainwindow, click on psfstyle
-> create default psfconfig of that style from a json that stores default parameters
-> create and store UIpsfconfig in frontend
-> create window using that
-> when done, retrieve new psfconfig from uipsfconfig and feed it to dolphin

PSF:

create psfgenerator inside dolphin or seperately? do they share libraries/functionalities? perhaps make seperate like cube but include here

see psfgenerator by the people that created deconvolutionlab2, this can also be used for the development of the psf gen algorithms

what to think about:
- is the psf generation computationally expensive?
- gpu?
- input/output format
- how will it be used? standalone?







