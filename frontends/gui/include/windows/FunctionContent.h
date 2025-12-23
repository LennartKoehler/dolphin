#pragma once
#include <functional>
#include "Window.h"

using function = std::function<void()>;

class FunctionContent : public Content{

public:
    FunctionContent(std::string name, function func);
    virtual void content() override;
    void setCallback(function func);
protected:
    function callback;
};

class ButtonContent : public FunctionContent{
public:
    ButtonContent(std::string name, function func);
    virtual void content() override;
};