# Debug Mode Rules - DOLPHIN Project

## Non-Obvious Debug Patterns

### Configuration Debugging
- Missing JSON fields throw `std::runtime_error` with exact field name
- Check config paths are relative to executable location, not project root  
- Deconvolution section must exist even when using algorithm defaults
- Use `config.printInfo = true` to display hyperstack metadata

### Algorithm Debug Issues
- Algorithm name mismatches cause runtime errors (case-sensitive)
- Factory registration errors appear as "Unknown algorithm" exceptions
- Check both factory registration AND CLI argument parsing match exactly
- GPU version requires CUDA_AVAILABLE compilation flag

### Memory Debugging Patterns
- PSFAllocator singleton issues not visible in stack traces
- Hyperstack copies use copy constructor - check for reference issues
- Shared_ptr cycles in PSFManager cause no visible crashes but prevent cleanup
- Large subimage arrays may cause memory fragmentation

### PSF-Specific Debug Issues
- PSF coordinate systems: GibsonLanni uses different conventions than Gaussian
- PSF safety border violations cause convolution artifacts at edges
- PSF array conflicts resolved by index order (lower index wins silently)
- PSF position arrays with duplicates use first occurrence

### GPU Debug Patterns
- CUDA build order: must build `lib/cube` before main project
- Runtime GPU errors indicate driver/cuda toolkit version mismatch
- GPU memory tracked separately from CPU - no unified memory debugging
- CUBE library compilation errors require separate debugging

### File I/O Debug Patterns
- TIFF detection based on extension: .tif, .tiff, .ometif supported
- Directory structure must match exactly for multi-file hyperstacks
- Results always saved to `../result/` relative to binary location
- Separate layer saving uses `sep` flag (not `seperate` typo)

## Performance Debugging

### OpenMP Issues
- Missing `-fopenmp` flag causes single-threaded performance
- Compiler optimization flags (-O3 -ffast-math) affect numerical precision
- Check thread count matches CPU core count for optimal performance

### Memory Performance
- Large subimage processing creates temporary memory copies
- FFTW uses separate memory pools - not visible in normal heap tracking
- Gaussian PSF generation temporarily uses 3x input image memory

### Algorithm Debug Issues
- Richardson-Lucy algorithms can become numerically unstable
- Total variation regularization requires careful lambda tuning
- GPU version shows 5-10x speedup but different numerical precision

## Hidden Gotchas

### Build Dependencies
- CUBE must be built before main project or compilation fails
- CUDA architecture hard-coded to "75;80;90" - unsupported GPUs fail silently
- OpenMP linking requires `-fopenmp` flag not present in standard CMAKE

### Configuration Edge Cases
- Negative numbers in JSON configs cause silent failures
- Missing algorithm names in JSON throw generic exceptions
- Zero-length arrays in PSF configs cause divide by zero errors
- Unicode paths in config cause file loading failures

### Platform-Specific Issues
- Linux build artifacts incompatible with Windows GPU libraries
- macOS Metal backend not implemented - OpenGL only
- Windows path separators work but project uses Unix separators

## Debug Workflow

### Configuration Debug Steps
1. Verify JSON paths relative to executable, not project root
2. Check all algorithm names match factory registration exactly
3. Ensure required fields exist - use default_config.json as template
4. Validate deconvolution section exists even if empty

### Algorithm Debug Steps  
1. Verify algorithm name spelling in factory registration
2. Check CLI argument parsing matches factory names
3. Validate algorithm-specific parameters in JSON
4. Test with RichardsonLucy (most stable) first

### Memory Debug Steps
1. Check for shared_ptr cycles in PSF/object relationships
2. Verify no uninitialized Hyperstack data before processing
3. Monitor memory usage with large subimage arrays
4. Watch for PSF safety border violations

### GPU Debug Steps
1. Verify CUDA toolkit version 12.1+ compatibility
2. Build CUBE library before main project
3. Check GPU architecture compatibility (75;80;90)
4. Test with simple algorithm before complex ones