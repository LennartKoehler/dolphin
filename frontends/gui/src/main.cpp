#include <dolphin/Dolphin.h>
#include "GUIFrontend.h"
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
    std::cout << "[Start Dolphin GUI]" << std::endl;
    
    // Initialize Dolphin with a proper log directory
    Dolphin* dolphin = new Dolphin();
    dolphin->init(resolveLogDir());
    
    // Create GUI frontend and run
    GUIFrontend* frontend = new GUIFrontend(dolphin);
    frontend->run();
    
    // Cleanup
    delete frontend;
    delete dolphin;
    
    std::cout << "[End Dolphin GUI]" << std::endl;
    return 0;
}


