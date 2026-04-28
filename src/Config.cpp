#include "dolphin/Config.h"

#include <spdlog/spdlog.h>


bool Config::loadFromJSON(const json& jsonData){
    bool success = true;
    try{
        logUnvalidParameters(jsonData);

        visitParams([this, &jsonData]<typename T>(T& value, ConfigParameter& param) {
            try{
                if (jsonData.contains(param.jsonTag)) {
                    value = jsonData.at(param.jsonTag).get<T>();
                }
            }
            catch (const std::exception& e){
                spdlog::get("config")->error("({}) Error in reading config from json: {}. Error occured in json parameter '{}'", this->getName(), e.what(), param.jsonTag);
                throw std::runtime_error("Config parameter error");
            }
        });
    }
    catch (const std::exception& e){
        success = false;
    }

    return success;
}

json Config::writeToJSON() const {
    json jsonData;
    const_cast<Config*>(this)->visitParams([&jsonData]<typename T>(const T& value, const ConfigParameter& param) {
        if (!param.value)
            return;
        jsonData[param.jsonTag] = value;

    });

    return jsonData;
}

// helper trait
namespace {
template<typename T, typename = void>
struct has_ostream : std::false_type {};

template<typename T>
struct has_ostream<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>> : std::true_type {};
}

void Config::printValues() const {
    const_cast<Config*>(this)->visitParams([this]<typename T>(const T& value, const ConfigParameter& param){
        auto logger = spdlog::get("config");

        if constexpr(std::is_same_v<T, std::vector<int>>) {
            // Build a joined string safely at runtime (no fmt instantiation for T)
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < value.size(); ++i) {
                if (i) oss << ", ";
                oss << value[i];
            }
            oss << "]";
            logger->debug("({}) {}: {}", this->getName(), param.name, oss.str());
            return;
        }
        else if constexpr(std::is_same_v<T, std::vector<std::string>>) {
            // Build a joined string safely at runtime (no fmt instantiation for T)
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < value.size(); ++i) {
                if (i) oss << ", ";
                oss << value[i];
            }
            oss << "]";
            logger->debug("({}) {}: {}", this->getName(), param.name, oss.str());
            return;
        }
        else if constexpr(std::is_same_v<T, std::array<int, 3>>) {
            // Build a joined string safely at runtime (no fmt instantiation for T)
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < value.size(); ++i) {
                if (i) oss << ", ";
                oss << value[i];
            }
            oss << "]";
            logger->debug("({}) {}: {}", this->getName(), param.name, oss.str());
            return;
        }

        // Non-vector parameters: use compile-time guards for formatting
        else if constexpr (fmt::is_formattable<T, char>::value) {
            logger->debug("({}) {}: {}", this->getName(), param.name, value);
        }
        else if constexpr (has_ostream<T>::value) {
            std::ostringstream oss;
            oss << value;
            logger->debug("({}) {}: {}", this->getName(), param.name, oss.str());
        }
        else {
            logger->debug("({}) {}: <unprintable type>", this->getName(), param.name);
        }
    });
}

json Config::loadJSONFile(const std::string& filePath) {
    std::ifstream configFile(filePath);
    if (!configFile.is_open()) {
        throw std::runtime_error("Could not open config file: " + filePath);
    }

    json jsonData;
    configFile >> jsonData;
    return jsonData;
}

bool Config::logUnvalidParameters(const json& jsonData) const {
    // First, collect all valid json tags from registered parameters
    std::unordered_set<std::string> validTags;
    for (const auto& param : parameters) {
        if (param.jsonTag) {
            validTags.insert(param.jsonTag);
        }
    }

    // Check for unknown keys in JSON
    std::vector<std::string> unknownKeys;
    for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
        if (validTags.find(it.key()) == validTags.end()) {
            unknownKeys.push_back(it.key());
        }
    }

    // Report unknown keys
    if (!unknownKeys.empty()) {
        for (const auto& key : unknownKeys) {
            spdlog::get("config")->warn("({}) Unknown key in config: {}", this->getName(), key);
        }

    }
    return true;
}

