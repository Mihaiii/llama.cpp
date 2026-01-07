set(TARGET_TTSV_TRAIN llama-ttsv-train)
set(TARGET_TTSV_RUN llama-ttsv-run)

add_executable(${TARGET_TTSV_TRAIN} ttsv-train.cpp)
add_executable(${TARGET_TTSV_RUN} ttsv-run.cpp)

foreach(target ${TARGET_TTSV_TRAIN} ${TARGET_TTSV_RUN})
    target_link_libraries(${target} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ../../src)
endforeach()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_TTSV_TRAIN} ${TARGET_TTSV_RUN} RUNTIME)
endif()
