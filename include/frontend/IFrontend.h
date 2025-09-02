#pragma once
#include "SetupConfig.h"

class IFrontend{
public:
    IFrontend(SetupConfig* config) : config(config){}
    virtual ~IFrontend(){}
    virtual void run() = 0;

protected:
    SetupConfig* config;
};