#pragma once
#include "DeconvolutionConfig.h"

class IFrontend{
public:
    IFrontend(ConfigManager* config) : config(config){}
    virtual ~IFrontend(){}
    virtual void run() = 0;

protected:
    ConfigManager* config;
};