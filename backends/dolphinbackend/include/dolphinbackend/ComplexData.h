/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once
#include <array>
#include <algorithm>
#include "CuboidShape.h"

class IBackendMemoryManager;

typedef float real_t;
typedef real_t complex_t[2];


class ComplexData{
public:
    complex_t* data;
    CuboidShape size;
    const IBackendMemoryManager* backend;

    // Take ownership of pre-allocated memory
    ComplexData() = default;
    ComplexData(const IBackendMemoryManager* b, complex_t* data, CuboidShape size);
    ~ComplexData();
    ComplexData(const ComplexData& other);
    ComplexData& operator=(const ComplexData& other);


    ComplexData(ComplexData&& other) noexcept;
    ComplexData& operator=(ComplexData&& other) noexcept;
};

