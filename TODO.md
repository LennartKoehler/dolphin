- change config setup, see last chatgpt chat, make configs not inheritance but composition hierarchy, then the visitor pattern can remain templated and just call visitor of parent
- should backends be singletons?

- connect grid processing to multithreading, so that number of owrker threads also dictates the gridsize, also perhaps include a maxMem flag


- implement cubepsfmap, do we even need it?

- ask zoltan about image extension and padding, why should i pad the psf? makes sense that the cubes are padded
- check if the deconvolutionalgorithms are correct




