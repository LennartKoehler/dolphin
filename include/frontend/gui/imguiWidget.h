#pragma once

#include "GUIStyleConfig.h"
#include <vector>

//basically my own api for common imgui calls
class imguiWidget{
public:
    imguiWidget() = default;
    virtual ~imguiWidget(){}
    virtual void display(const ParameterDescription& p) = 0;
    void operator() (const ParameterDescription& p);

};


class imguiSliderDouble : public imguiWidget{
public:
    void display(const ParameterDescription& p) override;
};

class imguiSliderInt : public imguiWidget {
public:
    void display(const ParameterDescription& p) override;
};

class imguiInputString : public imguiWidget {
public:
    void display(const ParameterDescription& p) override;
};

class imguiCheckbox : public imguiWidget {
public:
    void display(const ParameterDescription& p) override;
};

class imguiVectorInt : public imguiWidget {
public:
    void display(const ParameterDescription& p) override;

private:
    std::vector<int>* values = nullptr;
    int newValue = 0;  // Value to be added
    
    void displayElement(int* val, int index);
    void addElementWidget();
    void removeElement(int index);
};