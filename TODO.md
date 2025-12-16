
bug somewhere in the labeled deconvolution, somehow everything concerning the label is 0, the labels are correct and exist, the second psf is zero, the masks are all zero
should psflabelgroups be able to overlap?

Reader/Writer:
    maybe the reader can also keep the file open


    should the reader actually have a lot of padding logic? i think no but its much easier because it has all the dimensionality information and its easir to pass as padding something that goes out of bounds of image instead of passing e.g. as negative values. Also it can load data however it sees fit. i dont think the executor should know about imagemetadata and determine itself what part of the image can be read and what has to padded after having read the image. I still believe that the reader should not do padding, but think about this before implementing

    the tasks that are run should be in order of how data is read. So dont schedule all tasks, and then just run them, as later on they might not access contiguous cubes. Rather Schedule new tasks only when old tasks are done so that they are in order of how they were created which should also be in order of how they are loaded

    Think about imagePadding, if i want to cut images out of the image, the deconvstrategy epects the originalImage to already be padded. How do i pad the image without having to load everything. The coordinates of how the cubes are requested are also for a padded image, think about this. Pad individual strips if they are at the edge of the image. Think about how coordinates are translated from unpadded to padded image psf padding is just zeros, so this doesnt actually matter. 

    for the reader i think it should be fine to copy the cube out of the larger strip. This cube can then be edited in place by the thread working on that cube.

    reader/write for large images that dont fit on memory, streamline process. computationalplan and tasks can perhaps be created from sort of metdata of the image, then image reading/writing may be a part of the deconvolutionstrategy. Perhaps having its own reader and writer object that can also manage memory
        basic idea:
            read metadata of image, alls sizes etc, also read metadata of psf
            create copmutational plan and tasks based on that data aswell as the configs
            have a streamlined approach where perhaps one has reader and write threads, which replace current reader/writer, which actually directly read/write to the file, not just prepare data in memory.
            These threads should also have a cube loaded that can be processed by the worker threads
            perhaps it can be good to read multiple cubes at once from file do to their data being concurrent on disc. Perhaps for larger available memory one had more reader/writer threads which hold the cube on memory, but therefore guarantee that a cube is always available.
            Think about which core does what, perhaps each thread should do everything, so that when the data is read it might already be on l2 cache, but i dont think that matters as data is too large.
            Perhaps we also dont split reading/writing and processing on multiple threads, as there seems to be no use when basically everything can happen in parallel

    when reading image file that ends with .tif but the file doesnt acutally exist a nonintuitive error occurs

think about read/write threads especially for cpu applications, they should be on same core for better data transfer

someweird padding problems, causing sufficient padding to still produce boundary conditions
    somehow caused by padding for whatever reason, should have nothing to do with how data is read or written as when i init the result as zero the whole image is filled,
    for some reason half of the psf on both sides id not anough although mathematically it should be
    should also have nothing to do with normalization within the cube as i also tested that



split the deconvolutionalgorithms into init and run functions. the init function then has data allocations and stores those as member variables. this way the backends can do more specific tasks. e.g. the workerthreads perhaps shouldnt do data allocations that are performed within the algorithms -> for more optimization and overlap of data transfer and processing


think about how the rangemap for labelpsfmap is loaded. create a base loading function that will read from string, then create loading functions that will read from json obj, and or json file. think about how when and where the labelpsfmap should be loaded, i think it should be used in the deconvolutionstrategyfactory, and should also be loaded there and passed to the labeleddeconvolutionstrategy there, so that other strategies know nothing about it. on the other hand, should the factory really be doing loading? can i create a reader object that will handle all of that? perhaps one that can do tif reading, json reading etc. currently reading is all over the place, e.g. psfconfigs, where are those read? they "read themselves"


- think about how the different strategies are run. are parameters of the setupconfig or the deconvolutionconfig responsible for deciding this?
    so basically if setupconfig.labelPath not empty then create labeledDeconvolutionStrategy and do the labeledimagereading, perhaps the configure of labeledimagereading does the reading of the labeled image? on the other hand the setupconfig should stay at the deconvolutionservice level and not go into the deconvolutionstrategy. So maybe the deconvolutionconfig should have the path.
    Metaquestion: how do i get the two strategies to behave the same, while one needs the extra data loaded.perhaps the deconvolutionstrategyfactory should take the setupconfig as input, and based on that decide the strategy. And then also load the labeled image and labelpsfmap itself, so the deconolutionservice can treat the strategies the same and doesnt know there are differences.
    so the deconvolutionstrategyfactory gets the stupconfig and nothing else. Then it decides which strategy to create. if its labeleddeconv then it will read the labelpsfmap and the labelimage and pass it to the strategy (not through configure but through seperate setter function)
    then the configure can be exactly the same between both strategies, and the setupconfig values direct the path to take in the factory

- think about the labelpsfmap: i want to be able to provide a map (like rangemap) which maps label to psfid. this should somehow work for config and cli.
    do i need to be able to papss a map as string in cli or is it enough if i pass a filepath to a json which has the map? i think it should be possible to provide a map through cli: --labelPSFMap 0:5=[psfid1234, psfid5435],5:6=[...] --otherflags, this way if i only have a couple psfs, like 2 it should be possible to do this in cli. But it should also be possible to provide this within the config provided with -c aswell as as an extra parameter to cli that has a path to jsut the labelpsfmap


deconvolutionStrategy and processor refactor:

    deconvolutionstrategy has one thread pool, processor has its own worker pool, that is the divide, additionally deconvstrat should never use the backend (except to calc memory 
        usage maybe)
    deconvolutionstrategy runs the main loop, not processor.
    within the loop over cubes, strategy uses its own threadpool to get the cube, then, within that threadpool launches a task in the deconvolutionprocessor
        this will then use that thread to do some more copy operations using the backend.
    Then when the task is done (the worker thread goes back to processor) the strategy thread which launched it takes over again and copies it back. Here it uses its own stratgey to 
        copy back, so it might use its labels, it might not.
    This way also when launching tasks, the labeled strategy can use its already cut out cube, and loop over all labels in that cube
    homogeneouscubes (the literal cutting of the cubes) should just be a function or something else, not attached to the deconvolutinostrategy


- can i do blending without the regions overlapping (much easier)
- 
- currently setting number threads to max will be slower than e.g. 8/12 because I assume the overhead is larger and there is no longer a benefit to having a 
- add threads and memory restrictions to psfgenerator and the corresponding conifg
- create paddingStrategy? So that it can interact with the deconvolutionstrategy, to create ideal padded cubes
- change memory management to be only in the backend, and not in the deconvolutionprocessor
- can i get around using the cpumemorymanager? somehow directly convert from cv mat to fftw?








- make the maps for pre and postprocessing which indicate where each value comes from, dont actually copy though, as this is unneccessary. Then integrate these maps into the fftw_cvMat conversion, where a copy is inevitable -> LK do i actually care that much about there copies?



- only difference between dolphins implementation of gibsonlanni and big psf generator is it doesnt scale the max to 1


- check if the deconvolutionalgorithms are correct







questions to ask:


- do we need a flag for padding dimensions -> with padding smaller than psf there would be grid artifacts due to cyclic convolution


- psfs for different regions of the image:
    How do we know which psf is for which region?
    Can all regions (cubes) still have the same size (for faster fft)? Can they still be cuboids?
    Think about the difference between using theoretical psfs vs measured psfs
    
- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




