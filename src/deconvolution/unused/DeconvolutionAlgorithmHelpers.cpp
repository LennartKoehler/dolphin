#include "deconvolution/algorithms/DeconvolutionAlgorithmHelpers.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <mutex>
#include <atomic>

namespace DeconvolutionHelpers {

    // ================================
    // Static Member Definitions
    // ================================

    // Progress tracker static members
    std::string ProgressTracker::m_progressPattern = "\r[STATUS] Progress: {current}/{total} ({percentage:.1f}%)";
    std::mutex ProgressTracker::m_printMutex;

    // ================================
    // Memory Management Implementation
    // ================================

    MemoryTracker& MemoryTracker::getInstance() {
        static MemoryTracker instance;
        return instance;
    }

    void MemoryTracker::allocate(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_currentUsage.fetch_add(size, std::memory_order_relaxed);
        if (m_currentUsage.load(std::memory_order_relaxed) > m_peakUsage.load(std::memory_order_relaxed)) {
            m_peakUsage.store(m_currentUsage.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    }

    void MemoryTracker::deallocate(size_t size) {
        std::lock_guard<std::mutex> lock(m_mutex);
        size_t current = m_currentUsage.load(std::memory_order_relaxed);
        if (size > current) {
            m_currentUsage.store(0, std::memory_order_relaxed);
        } else {
            m_currentUsage.fetch_sub(size, std::memory_order_relaxed);
        }
    }

    size_t MemoryTracker::getCurrentUsage() const {
        return m_currentUsage.load(std::memory_order_relaxed);
    }

    size_t MemoryTracker::getPeakUsage() const {
        return m_peakUsage.load(std::memory_order_relaxed);
    }

    void MemoryTracker::reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_currentUsage.store(0, std::memory_order_relaxed);
        m_peakUsage.store(0, std::memory_order_relaxed);
    }

    void MemoryTracker::printMemoryUsage(const std::string& operation) const {
        size_t current = m_currentUsage.load(std::memory_order_relaxed);
        size_t peak = m_peakUsage.load(std::memory_order_relaxed);
        
        std::string currentStr = formatBytesStatic(current);
        std::string peakStr = formatBytesStatic(peak);
        
        std::lock_guard<std::mutex> lock(m_printMutex);
        std::cout << "[MEM] " << operation
                  << " - Current: " << currentStr
                  << ", Peak: " << peakStr << std::endl;
    }

    std::string MemoryTracker::formatBytesStatic(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unitIndex = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unitIndex < 3) {
            size /= 1024.0;
            unitIndex++;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
        return oss.str();
    }

    // ================================
    // Configuration and Setup Helpers Implementation
    // ================================

    ValidationResult validateConfiguration(
        int iterations,
        double epsilon,
        int subimageSize,
        int psfSafetyBorder,
        int cubeSize,
        bool useGrid,
        int borderType
    ) {
        ValidationResult result;
        
        // Validate iterations
        if (iterations < 1) {
            result.addError("Iterations must be at least 1");
        } else if (iterations > 1000) {
            result.addWarning("Iterations > 1000 may cause performance issues");
        }
        
        // Validate epsilon
        if (epsilon <= 0) {
            result.addError("Epsilon must be positive");
        } else if (epsilon > 1.0) {
            result.addWarning("Epsilon > 1.0 may cause numerical instability");
        }
        
        // Validate subimage size
        if (subimageSize < 0) {
            result.addError("Subimage size cannot be negative");
        }
        
        // Validate PSF safety border
        if (psfSafetyBorder < 0) {
            result.addError("PSF safety border cannot be negative");
        } else if (psfSafetyBorder > 100) {
            result.addWarning("PSF safety border > 100 may cause excessive padding");
        }
        
        // Validate cube size
        if (cubeSize < 0) {
            result.addError("Cube size cannot be negative");
        } else if (cubeSize > 1024) {
            result.addWarning("Cube size > 1024 may cause memory issues");
        } else if (cubeSize > 0 && cubeSize % 32 != 0) {
            result.addWarning("Cube size should be multiple of 32 for optimal FFT performance");
        }
        
        // Validate border type
        if (borderType < 0) {
            result.addError("Border type cannot be negative");
        }
        
        return result;
    }

    void logConfigurationSummary(
        int iterations,
        double epsilon,
        int subimageSize,
        int psfSafetyBorder,
        int cubeSize,
        bool useGrid,
        int borderType,
        const std::string& algorithmName
    ) {
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        if (!algorithmName.empty()) {
            std::cout << "Algorithm: " << algorithmName << std::endl;
        }
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Epsilon: " << epsilon << std::endl;
        std::cout << "Subimage Size: " << (subimageSize > 0 ? std::to_string(subimageSize) : "auto") << std::endl;
        std::cout << "PSF Safety Border: " << psfSafetyBorder << std::endl;
        std::cout << "Cube Size: " << (cubeSize > 0 ? std::to_string(cubeSize) : "auto") << std::endl;
        std::cout << "Grid Processing: " << (useGrid ? "enabled" : "disabled") << std::endl;
        std::cout << "Border Type: " << borderType << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
    }

    int calculateOptimalCubeSize(
        int psfWidth,
        int psfHeight,
        int psfDepth,
        int imageWidth,
        int imageHeight,
        int imageDepth,
        int minCubeSize,
        int maxCubeSize
    ) {
        // Minimum cube size should be at least as large as the largest PSF dimension
        int minRequiredSize = std::max({psfWidth, psfHeight, psfDepth});
        
        // Add safety margin (25% larger than PSF)
        int recommendedSize = static_cast<int>(std::ceil(minRequiredSize * 1.25));
        
        // Ensure multiple of 32 for FFT performance
        recommendedSize = ((recommendedSize + 31) / 32) * 32;
        
        // Clamp to valid range
        return std::clamp(recommendedSize, minCubeSize, maxCubeSize);
    }

    ValidationResult validateImageAndPsfDimensions(
        int imageWidth,
        int imageHeight,
        int imageDepth,
        int psfWidth,
        int psfHeight,
        int psfDepth
    ) {
        ValidationResult result;
        
        // Validate image dimensions
        if (imageWidth <= 0 || imageHeight <= 0 || imageDepth <= 0) {
            result.addError("Image dimensions must be positive");
        }
        
        // Validate PSF dimensions
        if (psfWidth <= 0 || psfHeight <= 0 || psfDepth <= 0) {
            result.addError("PSF dimensions must be positive");
        }
        
        // Validate PSF is smaller than image
        if (psfWidth >= imageWidth) {
            result.addError("PSF width (" + std::to_string(psfWidth) + 
                           ") must be smaller than image width (" + std::to_string(imageWidth) + ")");
        }
        if (psfHeight >= imageHeight) {
            result.addError("PSF height (" + std::to_string(psfHeight) + 
                           ") must be smaller than image height (" + std::to_string(imageHeight) + ")");
        }
        if (psfDepth >= imageDepth) {
            result.addError("PSF depth (" + std::to_string(psfDepth) + 
                           ") must be smaller than image depth (" + std::to_string(imageDepth) + ")");
        }
        
        // Validate PSF dimensions are reasonable (odd numbers are preferred for PSFs)
        if (psfWidth % 2 == 0) {
            result.addWarning("PSF width is even, odd dimensions (e.g., 21, 41, 61) are recommended");
        }
        if (psfHeight % 2 == 0) {
            result.addWarning("PSF height is even, odd dimensions (e.g., 21, 41, 61) are recommended");
        }
        if (psfDepth % 2 == 0) {
            result.addWarning("PSF depth is even, odd dimensions (e.g., 5, 7, 9) are recommended");
        }
        
        return result;
    }

    // ================================
    // Progress Tracker Implementation
    // ================================

    ProgressTracker::ProgressTracker(size_t totalItems, bool printProgress)
        : m_totalItems(totalItems), m_printProgress(printProgress) {
        m_currentProgress.store(0);
    }

    void ProgressTracker::incrementProgress() {
        size_t oldProgress = m_currentProgress.fetch_add(1);
        if (m_printProgress && oldProgress < m_totalItems) {
            printProgress();
        }
    }

    void ProgressTracker::setProgress(size_t currentProgress) {
        m_currentProgress.store(std::min(currentProgress, m_totalItems));
        if (m_printProgress) {
            printProgress();
        }
    }

    double ProgressTracker::getProgressPercentage() const {
        if (m_totalItems == 0) return 100.0;
        return (static_cast<double>(m_currentProgress.load()) / static_cast<double>(m_totalItems)) * 100.0;
    }

    void ProgressTracker::printProgress() const {
        std::lock_guard<std::mutex> lock(m_printMutex);
        
        size_t current = m_currentProgress.load();
        if (current > m_totalItems) current = m_totalItems;
        
        double percentage = getProgressPercentage();
        
        // Replace progress pattern placeholders
        std::string progressText = m_progressPattern;
        
        // Simple string replacement (in production, consider using std::regex or proper formatting)
        size_t pos = progressText.find("{current}");
        if (pos != std::string::npos) {
            progressText.replace(pos, 9, std::to_string(current));
        }
        
        pos = progressText.find("{total}");
        if (pos != std::string::npos) {
            progressText.replace(pos, 7, std::to_string(m_totalItems));
        }
        
        pos = progressText.find("{percentage:.1f}");
        if (pos != std::string::npos) {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "%.1f", percentage);
            progressText.replace(pos, 13, buffer);
        }
        
        // Print without newline for continuous progress
        std::cout << progressText;
        std::cout.flush();
        
        // Complete progress at 100%
        if (percentage >= 100.0) {
            std::cout << std::endl;
        }
    }

    void ProgressTracker::setProgressPattern(const std::string& pattern) {
        std::lock_guard<std::mutex> lock(m_printMutex);
        m_progressPattern = pattern;
    }

    // Template functions are already defined inline in the header file

    // Template functions are already defined inline in the header file
} // namespace DeconvolutionHelpers