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
#include "RectangleShape.h"

class IBackendMemoryManager;

typedef double complex[2];


class ComplexData{
public:
    complex* data;
    RectangleShape size;
    const IBackendMemoryManager* backend;

    // Take ownership of pre-allocated memory
    ComplexData() = default;
    ComplexData(const IBackendMemoryManager* b, complex* data, RectangleShape size);
    ~ComplexData();
    ComplexData(const ComplexData& other);
    ComplexData& operator=(const ComplexData& other);


    ComplexData(ComplexData&& other) noexcept;
};

