#pragma once

#include <spdlog/async.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/base_sink.h>

#include <functional>
#include <mutex>
#include <string>

namespace Logging{

    // Callback type for frontend log integration.
    // Receives (log level, message string) for every log message.
    // Note: Called from spdlog's async worker thread — ensure thread safety
    // in your callback (e.g. dispatch to GUI thread if needed).
    using LogCallback = std::function<void(spdlog::level::level_enum level, const std::string& message)>;

    // Custom spdlog sink that forwards log messages to a frontend callback.
    class FrontendSink : public spdlog::sinks::base_sink<std::mutex> {
    public:
        FrontendSink() = default;
        FrontendSink(LogCallback callback) {
            setCallback(callback);
        }

        void setCallback(LogCallback callback) {
            std::lock_guard<std::mutex> lock(this->mutex_);
            callback_ = std::move(callback);
        }

    protected:
        void sink_it_(const spdlog::details::log_msg& msg) override {
            if (callback_) {
                callback_(msg.level, std::string(msg.payload.data(), msg.payload.size()));
            }
        }

        void flush_() override {}

    private:
        LogCallback callback_;
    };

    // Get the shared frontend sink instance (single instance across all TUs).
    inline std::shared_ptr<FrontendSink> getFrontendSink() {
        static auto sink = std::make_shared<FrontendSink>();
        return sink;
    }

    // Register a frontend callback to receive all log messages.
    // Can be called before or after init().
    //   - Before init(): callback is stored and will be used when loggers are created.
    //   - After init(): callback takes effect immediately on all existing loggers.
    inline void setFrontendLogCallback(LogCallback callback) {
        getFrontendSink()->setCallback(std::move(callback));
    }
    inline void resetFrontendLogCallback() {
        getFrontendSink()->setCallback(LogCallback{});
    }

    static void init(){
        static bool isInitialized;
        if (!isInitialized){

            isInitialized = true;
            spdlog::init_thread_pool(8192, 1);
            auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            bool truncate = true;
            auto debugLogSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("debug.log", truncate);
            debugLogSink->set_level(spdlog::level::trace);
            consoleSink->set_level(spdlog::level::warn);

            // Frontend sink — always included, no-op until a callback is set via setFrontendLogCallback()
            auto frontendSink = getFrontendSink();
            frontendSink->set_level(spdlog::level::trace);

            std::vector<spdlog::sink_ptr> sinks {consoleSink, debugLogSink, frontendSink};


            std::unique_ptr<spdlog::pattern_formatter> formatterConsole = std::make_unique<spdlog::pattern_formatter>(
                "[%^%l%$] [%n] [%d-%m-%Y %H:%M:%S] %v"
            );


            std::unique_ptr<spdlog::pattern_formatter> formatterLog = std::make_unique<spdlog::pattern_formatter>(
                "[%^%l%$] [%n] [%d-%m-%Y %H:%M:%S] %v"
            );

            consoleSink->set_formatter(std::move(formatterConsole));
            debugLogSink->set_formatter(std::move(formatterLog));


            std::shared_ptr<spdlog::async_logger> backendlogger = std::make_shared<spdlog::async_logger>("backend", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            std::shared_ptr<spdlog::async_logger> configlogger = std::make_shared<spdlog::async_logger>("config", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            std::shared_ptr<spdlog::async_logger> psflogger = std::make_shared<spdlog::async_logger>("psf", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            std::shared_ptr<spdlog::async_logger> deconvolutionlogger = std::make_shared<spdlog::async_logger>("deconvolution", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            std::shared_ptr<spdlog::async_logger> defaultlogger = std::make_shared<spdlog::async_logger>("default", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            std::shared_ptr<spdlog::async_logger> readerlogger = std::make_shared<spdlog::async_logger>("reader", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);
            std::shared_ptr<spdlog::async_logger> writerlogger = std::make_shared<spdlog::async_logger>("writer", sinks.begin(), sinks.end(), spdlog::thread_pool(), spdlog::async_overflow_policy::block);

            backendlogger->set_level(spdlog::level::trace);
            configlogger->set_level(spdlog::level::trace);
            psflogger->set_level(spdlog::level::trace);
            deconvolutionlogger->set_level(spdlog::level::trace);
            defaultlogger->set_level(spdlog::level::trace);
            readerlogger->set_level(spdlog::level::trace);
            writerlogger->set_level(spdlog::level::trace);


            backendlogger->flush_on(spdlog::level::trace);
            spdlog::register_logger(backendlogger);
            deconvolutionlogger->flush_on(spdlog::level::trace);
            spdlog::register_logger(deconvolutionlogger);
            configlogger->flush_on(spdlog::level::trace);
            spdlog::register_logger(configlogger);
            psflogger->flush_on(spdlog::level::trace);
            spdlog::register_logger(psflogger);
            readerlogger->flush_on(spdlog::level::trace);
            spdlog::register_logger(readerlogger);
            writerlogger->flush_on(spdlog::level::trace);
            spdlog::register_logger(writerlogger);
            defaultlogger->flush_on(spdlog::level::trace);
            spdlog::set_default_logger(defaultlogger);

        }


    }

}
