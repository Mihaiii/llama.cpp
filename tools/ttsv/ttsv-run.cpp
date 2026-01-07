#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama.h"
#include "sampling.h"
#include "ttsv-common.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        return "";
    }
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return contents;
}

void fill_batch_embd(llama_batch & batch, const std::vector<float> & embd, int32_t n_embd, llama_pos pos_base) {
    const int32_t n_tokens = embd.size() / n_embd;
    batch.n_tokens         = n_tokens;

    std::memcpy(batch.embd, embd.data(), embd.size() * sizeof(float));

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = pos_base + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
}

void fill_batch_tokens(llama_batch &                    batch,
                       const std::vector<llama_token> & tokens,
                       llama_pos                        pos_base,
                       bool                             logits_last) {
    const int32_t n_tokens = tokens.size();
    batch.n_tokens         = n_tokens;

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = pos_base + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = logits_last && (i == n_tokens - 1);
    }
}

std::string format_prompt(const common_chat_templates * chat_templates,
                          const common_params &         params,
                          const std::string &           user_prompt,
                          bool                          add_generation_prompt) {
    if (!params.enable_chat_template || chat_templates == nullptr) {
        return user_prompt;
    }

    common_chat_templates_inputs inputs;
    inputs.use_jinja = params.use_jinja;
    if (!params.system_prompt.empty()) {
        common_chat_msg sys_msg;
        sys_msg.role    = "system";
        sys_msg.content = params.system_prompt;
        inputs.messages.push_back(sys_msg);
    }
    common_chat_msg user_msg;
    user_msg.role    = "user";
    user_msg.content = user_prompt;
    inputs.messages.push_back(user_msg);
    inputs.add_generation_prompt = add_generation_prompt;

    return common_chat_templates_apply(chat_templates, inputs).prompt;
}

}  // namespace

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTSV)) {
        return 1;
    }

    if (params.n_gpu_layers != 0) {
        fprintf(stderr, "llama-ttsv-run: forcing CPU only (n_gpu_layers=0)\n");
        params.n_gpu_layers = 0;
    }

    std::string prompt = params.prompt;
    if (!params.prompt_file.empty()) {
        prompt = read_file(params.prompt_file);
    }

    if (prompt.empty()) {
        fprintf(stderr, "llama-ttsv-run: prompt is empty\n");
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    if (!llama_init) {
        fprintf(stderr, "llama-ttsv-run: failed to load model\n");
        return 1;
    }

    llama_context * ctx   = llama_init->context();
    llama_model *   model = llama_init->model();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "llama-ttsv-run: invalid model/context\n");
        return 1;
    }

    common_chat_templates_ptr chat_templates;
    if (params.enable_chat_template) {
        chat_templates = common_chat_templates_init(model, params.chat_template);
    }

    int32_t            prefix_len  = 0;
    int32_t            prefix_embd = 0;
    std::vector<float> prefix;
    if (!params.ttsv_path.empty()) {
        if (!ttsv::load_prefix(params.ttsv_path, prefix_len, prefix_embd, prefix)) {
            fprintf(stderr, "llama-ttsv-run: failed to load prefix from %s\n", params.ttsv_path.c_str());
            return 1;
        }
        if (prefix_embd != llama_model_n_embd_inp(model)) {
            fprintf(stderr, "llama-ttsv-run: prefix embedding dim mismatch\n");
            return 1;
        }
        if (params.ttsv_scale != 1.0f) {
            for (auto & v : prefix) {
                v *= params.ttsv_scale;
            }
        }
    }

    const int32_t     n_embd         = llama_model_n_embd_inp(model);
    // const int32_t     max_new_tokens = params.n_predict > 0 ? params.n_predict : 128 * 4;
    const int32_t     max_new_tokens = 128 * 2;
    const std::string formatted      = format_prompt(chat_templates.get(), params, prompt, true);
    auto              prompt_tokens  = common_tokenize(ctx, formatted, true, true);

    common_sampler_ptr sampler(common_sampler_init(model, params.sampling));
    common_sampler_reset(sampler.get());

    llama_memory_clear(ctx->get_memory(), true);

    llama_pos pos = 0;

    if (prefix_len > 0) {
        llama_batch batch = llama_batch_init(prefix_len, n_embd, 1);
        fill_batch_embd(batch, prefix, n_embd, pos);
        llama_decode(ctx, batch);
        llama_batch_free(batch);
        pos += prefix_len;
    }

    if (!prompt_tokens.empty()) {
        llama_batch batch = llama_batch_init(prompt_tokens.size(), 0, 1);
        fill_batch_tokens(batch, prompt_tokens, pos, true);
        llama_decode(ctx, batch);
        llama_batch_free(batch);
        pos += prompt_tokens.size();
    }

    const llama_token eos = llama_vocab_eos(llama_model_get_vocab(model));

    std::string output;
    for (int32_t i = 0; i < max_new_tokens; ++i) {
        if (llama_get_logits_ith(ctx, -1) == nullptr) {
            break;
        }
        const llama_token id = common_sampler_sample(sampler.get(), ctx, -1, true);
        common_sampler_accept(sampler.get(), id, true);

        if (id == eos) {
            break;
        }

        output += common_token_to_piece(ctx, id, params.special);

        llama_batch              batch = llama_batch_init(1, 0, 1);
        std::vector<llama_token> tok   = { id };
        fill_batch_tokens(batch, tok, pos, true);
        llama_decode(ctx, batch);
        llama_batch_free(batch);
        pos += 1;
    }

    fprintf(stdout, "%s", output.c_str());

    llama_backend_free();

    return 0;
}
