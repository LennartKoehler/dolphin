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

#include <dolphin/Dolphin.h>
#include "CLIFrontend.h"
#include <filesystem>
#include <cstdlib>

// Resolve the platform-specific log directory for Dolphin.
// XDG_DATA_HOME is respected on Linux; falls back to ~/.local/share/dolphin,
// and ultimately to /tmp/dolphin if the home dir is unavailable.
static std::filesystem::path resolveLogDir() {
    // 1. Check XDG_DATA_HOME
    if (const char* xdg = std::getenv("XDG_DATA_HOME")) {
        return std::filesystem::path(xdg) / "dolphin";
    }
    // 2. Fallback to ~/.local/share/dolphin
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".local" / "share" / "dolphin";
    }
    // 3. Last resort: /tmp/dolphin
    return "/tmp/dolphin";
}

int main(int argc, char** argv) {
    // Initialize Dolphin with a proper log directory
    Dolphin* dolphin = new Dolphin();
    dolphin->init(resolveLogDir());

    // Create CLI frontend and run
    CLIFrontend frontend(dolphin, argc, argv);
    frontend.run();

    // Cleanup
    delete dolphin;
    return 0;
}
