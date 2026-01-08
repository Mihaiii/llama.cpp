#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama.h"
#include "sampling.h"
#include "ttsv-common.h"

#include <algorithm>
#include <cmath>
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

float control_vector_schedule_multiplier(const common_params & params, int32_t token_idx) {
    const bool enabled = params.control_vector_schedule_hold > 0 || params.control_vector_schedule_decay > 0 ||
                         params.control_vector_schedule_start != 1.0f || params.control_vector_schedule_end != 1.0f;
    if (!enabled) {
        return 1.0f;
    }

    if (token_idx < params.control_vector_schedule_hold) {
        return params.control_vector_schedule_start;
    }

    if (params.control_vector_schedule_decay <= 0) {
        return params.control_vector_schedule_end;
    }

    const int32_t t = token_idx - params.control_vector_schedule_hold;
    if (t >= params.control_vector_schedule_decay) {
        return params.control_vector_schedule_end;
    }

    const float frac = static_cast<float>(t) / static_cast<float>(params.control_vector_schedule_decay);
    return params.control_vector_schedule_start +
           (params.control_vector_schedule_end - params.control_vector_schedule_start) * frac;
}

float ttsv_logit_blend_schedule(const common_params & params, int32_t token_idx) {
    const bool enabled = params.ttsv_logit_blend_hold > 0 || params.ttsv_logit_blend_decay > 0 ||
                         params.ttsv_logit_blend_start != 1.0f || params.ttsv_logit_blend_end != 1.0f;
    if (!enabled) {
        return params.ttsv_logit_blend;
    }

    if (token_idx < params.ttsv_logit_blend_hold) {
        return params.ttsv_logit_blend_start;
    }

    if (params.ttsv_logit_blend_decay <= 0) {
        return params.ttsv_logit_blend_end;
    }

    const int32_t t = token_idx - params.ttsv_logit_blend_hold;
    if (t >= params.ttsv_logit_blend_decay) {
        return params.ttsv_logit_blend_end;
    }

    const float frac = static_cast<float>(t) / static_cast<float>(params.ttsv_logit_blend_decay);
    return params.ttsv_logit_blend_start + (params.ttsv_logit_blend_end - params.ttsv_logit_blend_start) * frac;
}

float entropy_from_logits(const float * logits, int n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        return 0.0f;
    }

    double max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum_exp       = 0.0;
    double sum_exp_logit = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        const double v = std::exp(static_cast<double>(logits[i]) - max_logit);
        sum_exp += v;
        sum_exp_logit += v * logits[i];
    }
    const double logsum     = std::log(sum_exp) + max_logit;
    const double mean_logit = sum_exp_logit / sum_exp;
    const double entropy    = logsum - mean_logit;
    return static_cast<float>(entropy);
}

bool apply_control_vector_scale(llama_context *                    ctx,
                                const common_control_vector_data & cvec,
                                int32_t                            layer_start,
                                int32_t                            layer_end,
                                float                              scale,
                                std::vector<float> &               scratch) {
    if (ctx == nullptr || cvec.n_embd <= 0 || cvec.data.empty()) {
        return false;
    }

    if (scratch.size() != cvec.data.size()) {
        scratch.resize(cvec.data.size());
    }
    for (size_t i = 0; i < cvec.data.size(); ++i) {
        scratch[i] = cvec.data[i] * scale;
    }

    const int err = llama_apply_adapter_cvec(ctx, scratch.data(), scratch.size(), cvec.n_embd, layer_start, layer_end);
    return err == 0;
}

llama_token sample_from_logits(common_sampler * sampler, const float * logits, std::vector<llama_token_data> & cur) {
    for (size_t i = 0; i < cur.size(); ++i) {
        cur[i] = llama_token_data{ static_cast<llama_token>(i), logits[i], 0.0f };
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(common_sampler_get(sampler), &cur_p);

    if (cur_p.selected < 0) {
        cur_p.selected = 0;
        for (size_t i = 1; i < cur.size(); ++i) {
            if (cur[i].logit > cur[cur_p.selected].logit) {
                cur_p.selected = i;
            }
        }
    }

    return cur[cur_p.selected].id;
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

    const bool ttsv_blend_schedule_enabled = params.ttsv_logit_blend_hold > 0 || params.ttsv_logit_blend_decay > 0 ||
                                             params.ttsv_logit_blend_start != 1.0f ||
                                             params.ttsv_logit_blend_end != 1.0f;
    const float   ttsv_blend      = std::clamp(params.ttsv_logit_blend, 0.0f, 1.0f);
    const bool    use_logit_blend = ttsv_blend < 0.999f || ttsv_blend_schedule_enabled;
    common_params params_base     = params;
    if (use_logit_blend) {
        params_base.control_vectors.clear();
        params_base.control_vector_layer_start = -1;
        params_base.control_vector_layer_end   = -1;
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

    common_init_result_ptr llama_init_base;
    llama_context *        ctx_base = nullptr;
    if (use_logit_blend) {
        llama_init_base = common_init_from_params(params_base);
        if (!llama_init_base || llama_init_base->context() == nullptr || llama_init_base->model() == nullptr) {
            fprintf(stderr, "llama-ttsv-run: failed to load base model for logit blending\n");
            return 1;
        }
        ctx_base = llama_init_base->context();
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
    const int32_t     max_new_tokens = params.n_predict > 0 ? params.n_predict : 128 * 4;
    //const int32_t     max_new_tokens = 128 * 2;
    const std::string formatted      = format_prompt(chat_templates.get(), params, prompt, true);
    auto              prompt_tokens  = common_tokenize(ctx, formatted, true, true);

    common_sampler_ptr sampler(common_sampler_init(model, params.sampling));
    common_sampler_reset(sampler.get());

    const int                     n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token_data> sample_cur(n_vocab);
    std::vector<float>            blended_logits;
    if (use_logit_blend) {
        blended_logits.resize(n_vocab);
    }

    const bool schedule_enabled = params.control_vector_schedule_hold > 0 || params.control_vector_schedule_decay > 0 ||
                                  params.control_vector_schedule_start != 1.0f ||
                                  params.control_vector_schedule_end != 1.0f;
    const bool backoff_enabled = params.control_vector_entropy_floor > 0.0f &&
                                 params.control_vector_entropy_backoff_tokens > 0 &&
                                 params.control_vector_entropy_backoff_scale >= 0.0f;
    const bool                 cvec_dynamic = !params.control_vectors.empty() && (schedule_enabled || backoff_enabled);
    common_control_vector_data cvec;
    std::vector<float>         cvec_scaled;
    float                      current_cvec_scale = 1.0f;
    int                        backoff_remaining  = 0;
    const float                backoff_scale      = std::clamp(params.control_vector_entropy_backoff_scale, 0.0f, 1.0f);
    const int32_t cvec_layer_start = params.control_vector_layer_start > 0 ? params.control_vector_layer_start : 1;
    const int32_t cvec_layer_end =
        params.control_vector_layer_end > 0 ? params.control_vector_layer_end : llama_model_n_layer(model);

    if (cvec_dynamic) {
        cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            fprintf(stderr, "llama-ttsv-run: failed to load control vector(s)\n");
            return 1;
        }
    }

    llama_memory_clear(ctx->get_memory(), true);
    if (ctx_base != nullptr) {
        llama_memory_clear(ctx_base->get_memory(), true);
    }

    llama_pos pos      = 0;
    llama_pos pos_base = 0;

    if (cvec_dynamic) {
        current_cvec_scale = control_vector_schedule_multiplier(params, 0);
        if (!apply_control_vector_scale(ctx, cvec, cvec_layer_start, cvec_layer_end, current_cvec_scale, cvec_scaled)) {
            fprintf(stderr, "llama-ttsv-run: failed to apply control vector schedule\n");
            return 1;
        }
    }

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

        if (ctx_base != nullptr) {
            llama_batch batch_base = llama_batch_init(prompt_tokens.size(), 0, 1);
            fill_batch_tokens(batch_base, prompt_tokens, pos_base, true);
            llama_decode(ctx_base, batch_base);
            llama_batch_free(batch_base);
            pos_base += prompt_tokens.size();
        }
    }

    const llama_token eos = llama_vocab_eos(llama_model_get_vocab(model));

    std::string              output;
    std::vector<llama_token> generated;
    generated.reserve(max_new_tokens);
    int       collapse_hits     = 0;
    const int collapse_window   = params.ttsv_collapse_window;
    const int collapse_patience = params.ttsv_collapse_patience;
    for (int32_t i = 0; i < max_new_tokens; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            break;
        }
        const float * logits_base = nullptr;
        if (use_logit_blend) {
            logits_base = llama_get_logits_ith(ctx_base, -1);
            if (logits_base == nullptr) {
                break;
            }
        }

        if (cvec_dynamic && backoff_enabled) {
            const float entropy = entropy_from_logits(logits, n_vocab);
            if (entropy < params.control_vector_entropy_floor) {
                backoff_remaining = params.control_vector_entropy_backoff_tokens;
            }
        }

        llama_token id = LLAMA_TOKEN_NULL;
        if (use_logit_blend) {
            float blend_t = ttsv_blend;
            if (ttsv_blend_schedule_enabled) {
                blend_t = std::clamp(ttsv_logit_blend_schedule(params, i), 0.0f, 1.0f);
            }
            for (int t = 0; t < n_vocab; ++t) {
                blended_logits[t] = logits_base[t] + blend_t * (logits[t] - logits_base[t]);
            }
            id = sample_from_logits(sampler.get(), blended_logits.data(), sample_cur);
        } else {
            id = common_sampler_sample(sampler.get(), ctx, -1, true);
        }
        common_sampler_accept(sampler.get(), id, true);

        if (id == eos) {
            break;
        }

        output += common_token_to_piece(ctx, id, params.special);
        generated.push_back(id);

        // Stop early if we fall into a short token loop.
        if (collapse_window > 0 && collapse_patience > 0 && generated.size() >= static_cast<size_t>(collapse_window)) {
            int         unique = 0;
            llama_token first  = LLAMA_TOKEN_NULL;
            llama_token second = LLAMA_TOKEN_NULL;
            for (size_t j = generated.size() - collapse_window; j < generated.size(); ++j) {
                const llama_token tok = generated[j];
                if (unique == 0) {
                    first  = tok;
                    unique = 1;
                } else if (tok == first) {
                    continue;
                } else if (unique == 1) {
                    second = tok;
                    unique = 2;
                } else if (tok == second) {
                    continue;
                } else {
                    unique = 3;
                    break;
                }
            }

            if (unique <= 2) {
                collapse_hits++;
            } else {
                collapse_hits = 0;
            }

            if (collapse_hits >= collapse_patience) {
                break;
            }
        }

        if (cvec_dynamic) {
            float next_scale = control_vector_schedule_multiplier(params, i + 1);
            if (backoff_enabled && backoff_remaining > 0) {
                next_scale *= backoff_scale;
                backoff_remaining--;
            }
            if (std::fabs(next_scale - current_cvec_scale) > 1e-6f) {
                if (!apply_control_vector_scale(ctx, cvec, cvec_layer_start, cvec_layer_end, next_scale, cvec_scaled)) {
                    fprintf(stderr, "llama-ttsv-run: failed to apply control vector schedule\n");
                    return 1;
                }
                current_cvec_scale = next_scale;
            }
        }

        llama_batch              batch = llama_batch_init(1, 0, 1);
        std::vector<llama_token> tok   = { id };
        fill_batch_tokens(batch, tok, pos, true);
        llama_decode(ctx, batch);
        llama_batch_free(batch);
        pos += 1;

        if (ctx_base != nullptr) {
            llama_batch batch_base = llama_batch_init(1, 0, 1);
            fill_batch_tokens(batch_base, tok, pos_base, true);
            llama_decode(ctx_base, batch_base);
            llama_batch_free(batch_base);
            pos_base += 1;
        }
    }

    fprintf(stdout, "%s", output.c_str());

    llama_backend_free();

    return 0;
}
