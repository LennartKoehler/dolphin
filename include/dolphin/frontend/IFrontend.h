#pragma once

class Dolphin;
class IFrontend{
public:
    IFrontend(Dolphin* dolphin) : dolphin(dolphin){}
    virtual ~IFrontend(){}
    virtual void run() = 0;

protected:
    Dolphin* dolphin;
};