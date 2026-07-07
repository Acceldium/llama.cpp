# qwen3tts-lib: shared TTS pipeline library
# Used by both llama-qwen3tts CLI and llama-server (/v1/audio/speech endpoint)
set(QWEN3TTS_LIB qwen3tts-lib)
add_library(${QWEN3TTS_LIB} STATIC
    qwen3tts-lib.cpp
    qwen3tts-lib.h
)
target_include_directories(${QWEN3TTS_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(${QWEN3TTS_LIB} PUBLIC llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${QWEN3TTS_LIB} PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(${QWEN3TTS_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# llama-tts: OuteTTS CLI
set(TARGET llama-tts)
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# llama-qwen3tts: Qwen3-TTS CLI (thin wrapper around qwen3tts-lib)
# live_mic.cpp: --live-mic capture, using the miniaudio single-header lib already
# vendored for tools/mtmd (see vendor/miniaudio/miniaudio.h).
# silero_vad.cpp: ggml-native Silero VAD, optional upgrade from live_mic.cpp's
# default RMS-energy VAD (see --vad-model in live_mic.cpp / qwen3tts.cpp).
set(TARGET2 llama-qwen3tts)
add_executable(${TARGET2} qwen3tts.cpp live_mic.cpp live_mic.h silero_vad.cpp silero_vad.h)
target_include_directories(${TARGET2} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET2} PRIVATE ${QWEN3TTS_LIB} llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET2} PRIVATE cxx_std_17)
if (NOT MSVC)
    target_compile_options(${TARGET2} PRIVATE -Wno-cast-qual) # miniaudio.h
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET2} RUNTIME)
endif()

set(TARGET3 speaker-encoder)
add_executable(${TARGET3} speaker-encoder/speaker_encoder.cpp)
target_link_libraries(${TARGET3} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET3} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET3} RUNTIME)
endif()
