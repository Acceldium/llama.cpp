set(TARGET llama-sortformer)
add_executable(${TARGET} sortformer.cpp)
install(TARGETS ${TARGET} RUNTIME)
# Sortformer is not an LLM: it links ggml directly (gguf + backend + cpu), no libllama.
target_link_libraries(${TARGET} PRIVATE ggml Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
