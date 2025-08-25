#pragma once

class imguiDisplayer;


class DefaultGUIStyleConfig : public GUIStyleConfig{
    DefaultGUIStyleConfig();

    void drawParameter(const ParameterDescription& param) override;

private:
    void registerDisplays();
    std::unordered_map<ParameterType, std::unique_ptr<imguiDisplayer>> styleGuide;
};