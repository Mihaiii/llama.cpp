#pragma once

#include "llama.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ttsv {

struct header {
    char     magic[8];
    uint32_t version;
    uint32_t n_tokens;
    uint32_t n_embd;
};

inline bool save_prefix(const std::string & path, int32_t n_tokens, int32_t n_embd, const std::vector<float> & data) {
    header h{};
    std::memcpy(h.magic, "TTSV\0\0\0\0", 8);
    h.version  = 1;
    h.n_tokens = static_cast<uint32_t>(n_tokens);
    h.n_embd   = static_cast<uint32_t>(n_embd);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    out.write(reinterpret_cast<const char *>(&h), sizeof(h));
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    return out.good();
}

inline bool load_prefix(const std::string & path, int32_t & n_tokens, int32_t & n_embd, std::vector<float> & data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    header h{};
    in.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!in.good()) {
        return false;
    }

    if (std::memcmp(h.magic, "TTSV\0\0\0\0", 8) != 0 || h.version != 1) {
        return false;
    }

    n_tokens = static_cast<int32_t>(h.n_tokens);
    n_embd   = static_cast<int32_t>(h.n_embd);

    const size_t n = static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd);
    data.resize(n);
    in.read(reinterpret_cast<char *>(data.data()), n * sizeof(float));
    return in.good();
}

inline std::vector<std::string> load_prompts(const std::string & path) {
    std::vector<std::string> prompts;
    std::ifstream in(path);
    if (!in) {
        return prompts;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        prompts.push_back(line);
    }

    return prompts;
}

inline bool token_embeddings(const llama_model & model, const std::vector<llama_token> & tokens, std::vector<float> & out) {
    if (tokens.empty()) {
        out.clear();
        return true;
    }

    const int64_t n_embd = llama_model_n_embd_inp(&model);
    const int64_t n_tokens = tokens.size();

    const size_t size_meta = 2 * 1024 * 1024;
    ggml_init_params params = {
        /*.mem_size   =*/ size_meta,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(ids);

    ggml_tensor * embd = ggml_get_rows(ctx, model.tok_embd, ids);
    if (embd->type != GGML_TYPE_F32) {
        embd = ggml_cast(ctx, embd, GGML_TYPE_F32);
    }
    ggml_set_output(embd);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, embd);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(ids, tokens.data(), 0, tokens.size() * sizeof(llama_token));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    out.resize(static_cast<size_t>(n_embd * n_tokens));
    ggml_backend_tensor_get(embd, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return true;
}

} // namespace ttsv
