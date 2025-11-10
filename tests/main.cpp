#include <iostream>

// Function declarations from other files
void exampleMemoryOperation();
int main() {
    std::cout << "=== Running DOLPHIN Tests ===" << std::endl;
    
    // Run memory exception test
    std::cout << "\n=== Testing Memory Exception Handling ===" << std::endl;
    exampleMemoryOperation();
    


    
    std::cout << "\n=== All tests completed ===" << std::endl;
    return 0;
}