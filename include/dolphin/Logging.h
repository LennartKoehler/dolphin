#pragma once

#include <spdlog/async.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace Logging{

    static void init(){
        static bool isInitialized;
        if (!isInitialized){

            isInitialized = true;
            spdlog::init_thread_pool(8192, 1);
            auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            bool truncate = true;
            auto debugLogSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("debug.log", truncate);
            std::vector<spdlog::sink_ptr> sinks {consoleSink, debugLogSink};
            debugLogSink->set_level(spdlog::level::trace);
            consoleSink->set_level(spdlog::level::warn);

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


            defaultlogger->flush_on(spdlog::level::trace);  // so that we have logs even if it crashes

            spdlog::register_logger(backendlogger);
            spdlog::register_logger(deconvolutionlogger);
            spdlog::register_logger(configlogger);
            spdlog::register_logger(psflogger);
            spdlog::set_default_logger(defaultlogger);
        }
        

    }

}