#include <opencv2/core.hpp>
#include <iostream>
#include "UtlGrid.h"

// Function to split image in grid with separate depth division
std::vector<std::vector<cv::Mat>> UtlGrid::split3DImageIntoCubes(const std::vector<cv::Mat>& volume, int gridDivision, int depthDivision) {
    // Get the dimensions of the 3D volume
    int depth = volume.size();
    if (depth == 0) return {}; // Handle empty volume
    int height = volume[0].rows;
    int width = volume[0].cols;

    // Calculate the size of the cubes in each dimension
    int cubeDepth = std::ceil(static_cast<double>(depth) / depthDivision);
    int cubeHeight = std::ceil(static_cast<double>(height) / gridDivision);
    int cubeWidth = std::ceil(static_cast<double>(width) / gridDivision);

    // Prepare the output vector of sub-volumes
    int totalCubes = gridDivision * gridDivision * depthDivision;
    std::vector<std::vector<cv::Mat>> subVolumes(totalCubes);

    // Split the volume into cubes
    for (int d = 0; d < depthDivision; ++d) {
        for (int h = 0; h < gridDivision; ++h) {
            for (int w = 0; w < gridDivision; ++w) {
                int index = d * gridDivision * gridDivision + h * gridDivision + w;

                int depthStart = d * cubeDepth;
                int heightStart = h * cubeHeight;
                int widthStart = w * cubeWidth;

                int depthEnd = std::min(depthStart + cubeDepth, depth);
                int heightEnd = std::min(heightStart + cubeHeight, height);
                int widthEnd = std::min(widthStart + cubeWidth, width);

                int currentDepth = depthEnd - depthStart;
                subVolumes[index].resize(currentDepth);

                for (int z = 0; z < currentDepth; ++z) {
                    subVolumes[index][z] = volume[depthStart + z](cv::Range(heightStart, heightEnd),
                                                                  cv::Range(widthStart, widthEnd)).clone();
                }
            }
        }
    }

    return subVolumes;
}

// Function to mergeCubes grid back to image with separate depth division
std::vector<cv::Mat> UtlGrid::mergeCubesInto3DImage(const std::vector<std::vector<cv::Mat>>& subVolumes, int gridDivision, int depthDivision, int originalDepth, int originalHeight, int originalWidth) {
    // Calculate the size of the cubes in each dimension
    int cubeDepth = std::ceil(static_cast<double>(originalDepth) / depthDivision);
    int cubeHeight = std::ceil(static_cast<double>(originalHeight) / gridDivision);
    int cubeWidth = std::ceil(static_cast<double>(originalWidth) / gridDivision);

    // Prepare the output volume
    std::vector<cv::Mat> volume(originalDepth);

    for (int i = 0; i < originalDepth; ++i) {
        volume[i] = cv::Mat::zeros(originalHeight, originalWidth, subVolumes[0][0].type());
    }

    // Merge the cubes back together
    for (int d = 0; d < depthDivision; ++d) {
        for (int h = 0; h < gridDivision; ++h) {
            for (int w = 0; w < gridDivision; ++w) {
                int index = d * gridDivision * gridDivision + h * gridDivision + w;

                int depthStart = d * cubeDepth;
                int heightStart = h * cubeHeight;
                int widthStart = w * cubeWidth;

                int depthEnd = std::min(depthStart + cubeDepth, originalDepth);
                int heightEnd = std::min(heightStart + cubeHeight, originalHeight);
                int widthEnd = std::min(widthStart + cubeWidth, originalWidth);

                for (int z = depthStart; z < depthEnd; ++z) {
                    int localDepthIndex = z - depthStart;
                    const cv::Mat& cubeSlice = subVolumes[index][localDepthIndex];
                    cubeSlice.copyTo(volume[z](cv::Range(heightStart, heightEnd), cv::Range(widthStart, widthEnd)));
                }
            }
        }
    }

    return volume;
}
void UtlGrid::getMinXYZ(const std::vector<std::vector<cv::Mat>>& split_vec,int& new_size_x, int& new_size_y, int& new_size_z){
    int minCols = INT_MAX;
    int minRows = INT_MAX;
    int minSize = INT_MAX;

    for (const auto &vec: split_vec) {
        for (const auto &mat: vec) {
            if (mat.cols < minCols) {
                minCols = mat.cols;
            }
            if (mat.rows < minRows) {
                minRows = mat.rows;
            }
        }
        if (vec.size() < minSize) {
            minSize = vec.size();
        }
    }

    new_size_x = minCols;
    new_size_y = minRows;
    new_size_z = minSize;
}

void UtlGrid::extendImage(std::vector<cv::Mat>& image3D, int& padding, int borderType){
    std::vector<cv::Mat> reversedImage3D = image3D; // Kopie erstellen
    std::reverse(reversedImage3D.begin(), reversedImage3D.end()); // Kopie umkehren

    // Den umgekehrten Vektor am Anfang des Originalvektors einfügen
    if(padding > image3D.size()){
        padding = image3D.size();
        std::cout << "Adjusted padding to " << padding << "(Depth)"<<std::endl;
    }
    if(padding > image3D[0].rows){
        padding = image3D[0].rows;
        std::cout << "Adjusted padding to " << padding << "(Heigth)"<<std::endl;

    }
    if(padding > image3D[0].cols){
        padding = image3D[0].cols;
        std::cout << "Adjusted padding to " << padding << "(Width)"<<std::endl;

    }

// Ensure we insert the correct range of elements from reversedImage3D
    if (padding > 0 && padding <= reversedImage3D.size()) {
        image3D.insert(image3D.begin(), reversedImage3D.end() - padding, reversedImage3D.end());
    }
    image3D.insert(image3D.end(), reversedImage3D.begin(), reversedImage3D.begin() + padding);

    for(auto& layer : image3D){
        // Variablen für die Padding-Größen
        int top = padding;     // Obere Padding-Größe
        int bottom = padding;  // Untere Padding-Größe
        int left = padding;    // Linke Padding-Größe
        int right = padding;   // Rechte Padding-Größe
        // Padding hinzufügen mit Spiegelung
        cv::copyMakeBorder(layer, layer, top, bottom, left, right, borderType);
    }
}

std::vector<std::vector<cv::Mat>> UtlGrid::splitWithoutCubePadding(std::vector<cv::Mat>& image3D, int cubeSize, int padding){
    int totalCubeNumZ = (image3D.size()/cubeSize) * (image3D[0].cols/cubeSize) * (image3D[0].rows/cubeSize);

    // Calculate dimensions after applying padding
    int depthWithPadding = image3D.size() - 2 * padding;
    int heightWithPadding = image3D[0].rows - 2 * padding;
    int widthWithPadding = image3D[0].cols - 2 * padding;

    // Ensure dimensions are valid after padding
    if (depthWithPadding < 0 || heightWithPadding < 0 || widthWithPadding < 0) {
        throw std::invalid_argument("Padding is too large, resulting in non-positive dimensions.");
    }
    // Adjust for the case when dimension with padding equals zero
    if (depthWithPadding == 0) depthWithPadding = padding;
    if (heightWithPadding == 0) heightWithPadding = padding;
    if (widthWithPadding == 0) widthWithPadding = padding;

    std::vector<std::vector<cv::Mat>> split;

    for (int depth = padding; depth < padding + depthWithPadding; depth += cubeSize) {
        for (int width = padding; width < padding + widthWithPadding; width += cubeSize) {
            for (int height = padding; height < padding + heightWithPadding; height += cubeSize) {
                std::vector<cv::Mat> cube;
                // Collect slices in each cube
                for (int z = depth; z < depth + cubeSize; ++z) {
                    if(z >= image3D.size()){
                        cv::Mat emptySlice(cubeSize, cubeSize, image3D[0].type(), cv::Scalar(0));
                        cube.push_back(emptySlice);
                        continue;
                    }
                    // Calculate the dimensions for the current slice
                    int widthEnd = std::min(width + cubeSize, padding + widthWithPadding);
                    int heightEnd = std::min(height + cubeSize, padding + heightWithPadding);

                    // Define the rectangle to be cut out
                    cv::Rect cubeSlice(width, height, widthEnd - width, heightEnd - height);

                    // If the current slice is smaller than the cubeSize, add borders to fill it
                    if (widthEnd - width < cubeSize || heightEnd - height < cubeSize) {
                        cv::Mat paddedSlice;
                        int rightBorder = cubeSize - (widthEnd - width);
                        int bottomBorder = cubeSize - (heightEnd - height);

                        // Pad the image slice with zeros (black padding)
                        cv::copyMakeBorder(image3D[z](cubeSlice), paddedSlice, 0, bottomBorder, 0, rightBorder, cv::BORDER_CONSTANT, cv::Scalar(0));
                        cube.push_back(paddedSlice);
                    } else {
                        // Add the slice to the cube without padding
                        cube.push_back(image3D[z](cubeSlice));


                    }
                }
                // Fill in the rest of the Z dimension with empty (black) images if necessary
                if (cube.size() < cubeSize) {

                }
                split.push_back(cube);
            }
        }
    }

    return split;
}

std::vector<std::vector<cv::Mat>> UtlGrid::splitWithCubePadding(std::vector<cv::Mat>& image3D, int cubeSize, int padding, int cubePadding){
    int totalCubeNumZ = (image3D.size() / cubeSize) * (image3D[0].cols / cubeSize) * (image3D[0].rows / cubeSize);
// Calculate dimensions after applying padding
    int depthWithPadding = image3D.size() - 2 * padding;
    int heightWithPadding = image3D[0].rows - 2 * padding;
    int widthWithPadding = image3D[0].cols - 2 * padding;

    // Ensure dimensions are valid after padding
    if (depthWithPadding < 0 || heightWithPadding < 0 || widthWithPadding < 0) {
        throw std::invalid_argument("Padding is too large, resulting in non-positive dimensions.");
    }
    // Adjust for the case when dimension with padding equals zero
    if (depthWithPadding == 0) depthWithPadding = padding;
    if (heightWithPadding == 0) heightWithPadding = padding;
    if (widthWithPadding == 0) widthWithPadding = padding;

    std::vector<std::vector<cv::Mat>> split;

    for (int depth = padding; depth < padding + depthWithPadding; depth += cubeSize) {
        for (int width = padding; width < padding + widthWithPadding; width += cubeSize) {
            for (int height = padding; height < padding + heightWithPadding; height += cubeSize) {
                std::vector<cv::Mat> cube;

                for (int z = depth - cubePadding; z < depth + cubeSize + cubePadding; ++z) {
                    if (z >= image3D.size() || z < 0) {
                        cv::Mat emptySlice(cubeSize + 2 * cubePadding, cubeSize + 2 * cubePadding, image3D[0].type(), cv::Scalar(0));
                        cube.push_back(emptySlice);
                        continue;
                    }

                    int widthStart = std::max(0, width - cubePadding);
                    int heightStart = std::max(0, height - cubePadding);
                    int widthEnd = std::min(image3D[0].cols, width + cubeSize + cubePadding);
                    int heightEnd = std::min(image3D[0].rows, height + cubeSize + cubePadding);

                    cv::Rect cubeSlice(widthStart, heightStart, widthEnd - widthStart, heightEnd - heightStart);

                    cv::Mat paddedSlice;
                    int rightBorder = (width + cubeSize + cubePadding) - widthEnd;
                    int bottomBorder = (height + cubeSize + cubePadding) - heightEnd;
                    int leftBorder = widthStart - (width - cubePadding);
                    int topBorder = heightStart - (height - cubePadding);

                    cv::copyMakeBorder(image3D[z](cubeSlice), paddedSlice, topBorder, bottomBorder, leftBorder, rightBorder, cv::BORDER_CONSTANT, cv::Scalar(0));
                    cube.push_back(paddedSlice);
                }

                split.push_back(cube);
            }
        }
    }

    return split;
}

void UtlGrid::cropCubePadding(std::vector<std::vector<cv::Mat>>& split, int cubePadding){
    for (auto& cube : split) {
        // Entferne die ersten und letzten Layer entsprechend cubePadding
        if (cube.size() > 2 * cubePadding) {
            cube.erase(cube.begin(), cube.begin() + cubePadding);
            cube.erase(cube.end() - cubePadding, cube.end());
        } else {
            // Falls cube weniger Layer als 2 * cubePadding hat, alles entfernen
            cube.clear();
        }

        // Beschnitt der restlichen Layer in der X- und Y-Dimension
        for (auto& slice : cube) {
            int newWidth = slice.cols - 2 * cubePadding;
            int newHeight = slice.rows - 2 * cubePadding;

            // Sicherstellen, dass die neuen Dimensionen gültig sind
            if (newWidth > 0 && newHeight > 0) {
                cv::Rect cropRegion(cubePadding, cubePadding, newWidth, newHeight);
                slice = slice(cropRegion).clone();
            } else {
                // Wenn die neuen Dimensionen ungültig sind, eine leere Scheibe erstellen
                slice = cv::Mat(newHeight > 0 ? newHeight : 1, newWidth > 0 ? newWidth : 1, slice.type(), cv::Scalar(0));
            }
        }
    }
}

std::vector<cv::Mat> UtlGrid::mergeCubes(const std::vector<std::vector<cv::Mat>>& cubes, int imageWidth, int imageHeight, int imageDepth, int cubeSize) {
    // Create an empty 3D image with the given dimensions
    // std::vector<cv::Mat> image3D(imageDepth, cv::Mat(imageHeight, imageWidth, cubes[0][0].type(), cv::Scalar(0)));
    // Create an empty 3D image with the given dimensions
    // Cubes in this function already cropped and have no padding
    std::vector<cv::Mat> image3D;
    for (int i = 0; i < imageDepth; ++i) {
        image3D.push_back(cv::Mat(imageHeight, imageWidth, cubes[0][0].type(), cv::Scalar(0)));
    }
    int cubeIndex = 0; // Initialize the cube index to keep track of which cube to process

    // Iterate over the depth of the 3D image in steps of cubeSize
    for (int z = 0; z < imageDepth; z += cubeSize) {
        // Iterate over the width of the 3D image in steps of cubeSize
        for (int x = 0; x < imageWidth; x += cubeSize) {
            // Iterate over the height of the 3D image in steps of cubeSize
            for (int y = 0; y < imageHeight; y += cubeSize) {
                // Ensure there are enough cubes to fill the 3D image
                if (cubeIndex >= cubes.size()) {
                    throw std::runtime_error("Not enough cubes to fill the 3D image");
                }

                // Get the current cube to be processed
                const std::vector<cv::Mat>& cube = cubes[cubeIndex++];

                // Iterate over the depth slices of the current cube
                for (int dz = 0; dz < cubeSize && z + dz < imageDepth; ++dz) {
                    // Calculate the dimensions for the current slice
                    int widthEnd = std::min(x + cubeSize, imageWidth);
                    int heightEnd = std::min(y + cubeSize, imageHeight);

                    // Define the rectangle to be merged
                    cv::Rect cubeSliceRect(0, 0, widthEnd - x, heightEnd - y);

                    // Define the destination ROI in the 3D image
                    cv::Rect destRect(x, y, widthEnd - x, heightEnd - y);

                    // Debugging print statements to check values
                    /* std::cout << "z + dz: " << z + dz << " (depth layer in image3D), ";
                     std::cout << "cubeIndex: " << cubeIndex - 1 << ", ";
                     std::cout << "dz: " << dz << ", ";
                     std::cout << "destinationROI: (" << x << ", " << y << ", " << widthEnd - x << ", " << heightEnd - y << ")" << std::endl;*/

                    // Merge the cube slice into the 3D image
                    cv::Mat destinationROI = image3D[z + dz](destRect);
                    cube[dz](cubeSliceRect).copyTo(destinationROI);

                    // Debugging: Visual check (commented out for batch testing)
                    //cv::imshow("Current Cube Slice", cube[dz](cubeSliceRect));
                    // cv::imshow("Destination ROI Slice", destinationROI);
                    // cv::waitKey(100); // Adjust the delay as needed for debugging
                }
            }
        }
    }

    // Show final layers of the 3D image for verification
    /* for (int i = 0; i < image3D.size(); ++i) {
         std::cout << "Displaying layer " << i << std::endl;
         cv::imshow("Image Layer " + std::to_string(i), image3D[i]);
         cv::waitKey(500); // Adjust the delay as needed for verification
     }*/

    return image3D;
}

void UtlGrid::adjustCubeOverlap(std::vector<std::vector<cv::Mat>>& cubes, int cubePadding) {
    // Iterate through all cubes in the split set
    for (size_t cubeIndex = 0; cubeIndex < cubes.size(); ++cubeIndex) {
        std::vector<cv::Mat>& cube = cubes[cubeIndex];

        // Check if there is padding to adjust
        if (cubePadding <= 0) continue;

        // Adjust overlap for each depth slice in the cube
        for (int dz = 0; dz < cube.size(); ++dz) {
            cv::Mat& slice = cube[dz];

            // Define overlap regions in X and Y directions
            int overlapStart = cubePadding;
            int overlapEndX = slice.cols - cubePadding;
            int overlapEndY = slice.rows - cubePadding;

            // Iterate through the overlap region and adjust intensities
            for (int x = overlapStart; x < overlapEndX; ++x) {
                for (int y = overlapStart; y < overlapEndY; ++y) {
                    // Calculate the blending factor based on the distance to the edge
                    float alphaX = static_cast<float>(x - overlapStart) / cubePadding;
                    float alphaY = static_cast<float>(y - overlapStart) / cubePadding;
                    float alpha = std::min(alphaX, alphaY);

                    // Get the current intensity value
                    float currentValue = slice.at<float>(y, x);

                    // Adjust the intensity using the blending factor
                    slice.at<float>(y, x) = alpha * currentValue + (1 - alpha) * slice.at<float>(y, x);
                }
            }
        }
    }
}

