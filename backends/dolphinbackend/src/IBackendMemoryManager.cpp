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

#include "dolphinbackend/IBackendMemoryManager.h"

// Out-of-line virtual destructor to ensure vtable and typeinfo are generated
IBackendMemoryManager::~IBackendMemoryManager() = default;

// Out-of-line implementation of allocateMemoryOnDevice
// This ensures the vtable is properly emitted in this translation unit
void* IBackendMemoryManager::allocateMemoryOnDevice(size_t size) const {
    NOT_IMPLEMENTED(allocateMemoryOnDevice);
}
