#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama.h"
#include "sampling.h"
#include "ttsv-common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct style_pair_tokens {
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> response_tokens;
};

struct adamw_state {
    std::vector<float> m;
    std::vector<float> v;
    int64_t            step = 0;
};

float linear_lr(float lr0, float lr_min, int64_t step, int64_t total_steps) {
    if (total_steps <= 1) {
        return lr0;
    }
    const float t = float(step - 1) / float(total_steps - 1);
    return lr0 + (lr_min - lr0) * t;
}

void adamw_step(std::vector<float> &       params,
                const std::vector<float> & grad,
                adamw_state &              st,
                float                      lr,
                float                      beta1,
                float                      beta2,
                float                      eps,
                float                      wd) {
    st.step += 1;
    const float bias1 = 1.0f - std::pow(beta1, float(st.step));
    const float bias2 = 1.0f - std::pow(beta2, float(st.step));

    for (size_t i = 0; i < params.size(); ++i) {
        const float g = grad[i];
        st.m[i]       = beta1 * st.m[i] + (1.0f - beta1) * g;
        st.v[i]       = beta2 * st.v[i] + (1.0f - beta2) * g * g;

        const float m_hat = st.m[i] / bias1;
        const float v_hat = st.v[i] / bias2;

        params[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * params[i]);
    }
}

void init_prefix_random(std::vector<float> & prefix, int32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto & v : prefix) {
        v = dist(rng);
    }
}

void init_prefix_data_driven(std::vector<float> &       prefix,
                             int32_t                    n_embd,
                             const std::vector<float> & mean,
                             const std::vector<float> & stdev,
                             int32_t                    seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < prefix.size(); ++i) {
        const int32_t d = static_cast<int32_t>(i % n_embd);
        prefix[i]       = mean[d] + stdev[d] * dist(rng);
    }
}

std::vector<llama_token> build_token_set(llama_context * ctx, const std::vector<std::string> & pieces) {
    std::unordered_set<llama_token> uniq;
    for (const auto & piece : pieces) {
        const auto tokens = common_tokenize(ctx, piece, false, true);
        for (auto tok : tokens) {
            uniq.insert(tok);
        }
    }
    std::vector<llama_token> out;
    out.reserve(uniq.size());
    for (auto tok : uniq) {
        out.push_back(tok);
    }
    return out;
}

bool compute_mean_embedding(llama_context *                  ctx,
                            llama_model *                    model,
                            const std::vector<std::string> & texts,
                            std::vector<float> &             out) {
    const int32_t n_embd = llama_model_n_embd_inp(model);
    out.assign(n_embd, 0.0f);
    int64_t token_count = 0;

    for (const auto & text : texts) {
        const auto tokens = common_tokenize(ctx, text, false, true);
        if (tokens.empty()) {
            continue;
        }
        std::vector<float> embd;
        if (!ttsv::token_embeddings(*model, tokens, embd)) {
            continue;
        }
        for (size_t t = 0; t < tokens.size(); ++t) {
            const float * row = embd.data() + t * n_embd;
            for (int32_t d = 0; d < n_embd; ++d) {
                out[d] += row[d];
            }
        }
        token_count += static_cast<int64_t>(tokens.size());
    }

    if (token_count == 0) {
        return false;
    }

    const float inv = 1.0f / static_cast<float>(token_count);
    for (int32_t d = 0; d < n_embd; ++d) {
        out[d] *= inv;
    }
    return true;
}

bool should_data_init(const llama_model & model) {
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
            return true;
        default:
            return false;
    }
}

void fill_batch_embd(llama_batch &               batch,
                     const std::vector<float> &  embd,
                     int32_t                     n_embd,
                     llama_pos                   pos_base,
                     const std::vector<int8_t> & logits) {
    const int32_t n_tokens = embd.size() / n_embd;
    batch.n_tokens         = n_tokens;

    std::memcpy(batch.embd, embd.data(), embd.size() * sizeof(float));

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = pos_base + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = logits.empty() ? 0 : logits[i];
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

ggml_tensor * build_entropy_loss(ggml_context * ctx, ggml_tensor * logits, int64_t n_outputs) {
    ggml_tensor * probs     = ggml_soft_max(ctx, logits);
    ggml_tensor * log_probs = ggml_log(ctx, probs);
    ggml_tensor * mul       = ggml_mul(ctx, probs, log_probs);
    ggml_tensor * sum_vocab = ggml_sum_rows(ctx, mul);
    ggml_tensor * entropy   = ggml_scale(ctx, sum_vocab, -1.0f);
    ggml_tensor * loss      = ggml_sum(ctx, entropy);
    if (n_outputs > 0) {
        loss = ggml_scale(ctx, loss, 1.0f / float(n_outputs));
    }
    return loss;
}

std::vector<llama_token> generate_response(llama_context *                  ctx,
                                           const llama_model *              model,
                                           common_sampler *                 sampler,
                                           const std::vector<float> &       prefix,
                                           int32_t                          prefix_len,
                                           int32_t                          n_embd,
                                           const std::vector<llama_token> & prompt_tokens,
                                           int32_t                          max_new_tokens) {
    std::vector<llama_token> generated;

    llama_memory_clear(ctx->get_memory(), true);
    common_sampler_reset(sampler);

    llama_pos pos = 0;

    if (prefix_len > 0) {
        llama_batch         batch = llama_batch_init(prefix_len, n_embd, 1);
        std::vector<int8_t> logits(prefix_len, 0);
        logits.back() = 1;
        fill_batch_embd(batch, prefix, n_embd, pos, logits);
        if (llama_decode(ctx, batch) < 0) {
            llama_batch_free(batch);
            return generated;
        }
        llama_batch_free(batch);
        pos += prefix_len;
    }

    if (!prompt_tokens.empty()) {
        llama_batch batch = llama_batch_init(prompt_tokens.size(), 0, 1);
        fill_batch_tokens(batch, prompt_tokens, pos, true);
        if (llama_decode(ctx, batch) < 0) {
            llama_batch_free(batch);
            return generated;
        }
        llama_batch_free(batch);
        pos += prompt_tokens.size();
    } else if (prefix_len == 0) {
        return generated;
    }

    const llama_token eos = llama_vocab_eos(llama_model_get_vocab(model));

    for (int32_t i = 0; i < max_new_tokens; ++i) {
        if (llama_get_logits_ith(ctx, -1) == nullptr) {
            return generated;
        }
        const llama_token id = common_sampler_sample(sampler, ctx, -1, true);
        common_sampler_accept(sampler, id, true);
        if (id == eos) {
            break;
        }
        generated.push_back(id);

        llama_batch              batch = llama_batch_init(1, 0, 1);
        std::vector<llama_token> tok   = { id };
        fill_batch_tokens(batch, tok, pos, true);
        if (llama_decode(ctx, batch) < 0) {
            llama_batch_free(batch);
            return generated;
        }
        llama_batch_free(batch);
        pos += 1;
    }

    return generated;
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

std::vector<style_pair_tokens> load_style_pairs(llama_context *                 ctx,
                                                const common_chat_templates *   chat_templates,
                                                const common_params &           params,
                                                const std::string &             path) {
    std::vector<style_pair_tokens> out;
    if (path.empty()) {
        return out;
    }

    std::ifstream in(path);
    if (!in) {
        return out;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        const size_t sep = line.find('\t');
        if (sep == std::string::npos) {
            continue;
        }

        std::string prompt = line.substr(0, sep);
        std::string response = line.substr(sep + 1);
        if (prompt.empty() || response.empty()) {
            continue;
        }

        const std::string formatted = format_prompt(chat_templates, params, prompt, true);
        auto prompt_tokens = common_tokenize(ctx, formatted, true, true);

        response += "<|im_end|>";
        auto response_tokens = common_tokenize(ctx, response, false, true);

        if (prompt_tokens.empty() || response_tokens.empty()) {
            continue;
        }

        style_pair_tokens pair;
        pair.prompt_tokens = std::move(prompt_tokens);
        pair.response_tokens = std::move(response_tokens);
        out.push_back(std::move(pair));
    }

    return out;
}

bool compute_embedding_stats(llama_context *                  ctx,
                             llama_model *                    model,
                             const common_chat_templates *    chat_templates,
                             const common_params &            params,
                             const std::vector<std::string> & prompts,
                             std::vector<float> &             mean,
                             std::vector<float> &             stdev,
                             int64_t &                        token_count) {
    const int32_t      n_embd = llama_model_n_embd_inp(model);
    std::vector<float> sum(n_embd, 0.0f);
    std::vector<float> sum_sq(n_embd, 0.0f);
    token_count = 0;

    for (const auto & prompt : prompts) {
        const std::string formatted = format_prompt(chat_templates, params, prompt, true);
        const auto        tokens    = common_tokenize(ctx, formatted, true, true);
        if (tokens.empty()) {
            continue;
        }

        std::vector<float> embd;
        if (!ttsv::token_embeddings(*model, tokens, embd)) {
            continue;
        }

        const int64_t n_tokens = tokens.size();
        for (int64_t t = 0; t < n_tokens; ++t) {
            const float * row = embd.data() + t * n_embd;
            for (int32_t d = 0; d < n_embd; ++d) {
                const float v = row[d];
                sum[d] += v;
                sum_sq[d] += v * v;
            }
        }
        token_count += n_tokens;
    }

    if (token_count == 0) {
        return false;
    }

    mean.resize(n_embd);
    stdev.resize(n_embd);
    const float denom = 1.0f / static_cast<float>(token_count);
    for (int32_t d = 0; d < n_embd; ++d) {
        const float m = sum[d] * denom;
        const float v = std::max(0.0f, sum_sq[d] * denom - m * m);
        mean[d]       = m;
        stdev[d]      = std::sqrt(std::max(v, 1e-6f));
    }
    return true;
}

bool compute_supervised_nll_loss(llama_context *                  ctx,
                                 llama_model *                    model,
                                 const std::vector<float> &       prefix,
                                 int32_t                          prefix_len,
                                 int32_t                          n_embd,
                                 const std::vector<llama_token> & prompt_tokens,
                                 const std::vector<llama_token> & response_tokens,
                                 float &                          loss_out) {
    if (response_tokens.empty()) {
        return false;
    }
    if (prompt_tokens.empty() && prefix_len <= 0) {
        return false;
    }

    std::vector<llama_token> full_tokens = prompt_tokens;
    full_tokens.insert(full_tokens.end(), response_tokens.begin(), response_tokens.end());

    std::vector<float> token_embd;
    if (!ttsv::token_embeddings(*model, full_tokens, token_embd)) {
        return false;
    }

    std::vector<float> full_embd;
    full_embd.resize(prefix.size() + token_embd.size());
    std::memcpy(full_embd.data(), prefix.data(), prefix.size() * sizeof(float));
    std::memcpy(full_embd.data() + prefix.size(), token_embd.data(), token_embd.size() * sizeof(float));

    const int32_t       total_tokens = prefix_len + static_cast<int32_t>(full_tokens.size());
    std::vector<int8_t> logits_mask(total_tokens, 0);

    int32_t first_logit_pos = 0;
    if (!prompt_tokens.empty()) {
        first_logit_pos = prefix_len + static_cast<int32_t>(prompt_tokens.size()) - 1;
    } else {
        first_logit_pos = prefix_len - 1;
    }
    if (first_logit_pos < 0) {
        return false;
    }

    for (size_t i = 0; i < response_tokens.size(); ++i) {
        const int32_t pos = first_logit_pos + static_cast<int32_t>(i);
        if (pos < 0 || pos >= total_tokens) {
            return false;
        }
        logits_mask[pos] = 1;
    }

    llama_memory_clear(ctx->get_memory(), true);

    llama_batch batch = llama_batch_init(total_tokens, n_embd, 1);
    fill_batch_embd(batch, full_embd, n_embd, 0, logits_mask);

    llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
    if (!balloc.init(batch, model->vocab, ctx->get_memory(), n_embd, ctx->n_seq_max(), false)) {
        llama_batch_free(batch);
        return false;
    }

    auto mctx = ctx->get_memory()->init_batch(balloc, ctx->n_ubatch(), false);
    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        llama_batch_free(batch);
        return false;
    }

    ggml_status status = GGML_STATUS_SUCCESS;
    auto *      res    = ctx->build_graph_for_batch(mctx->get_ubatch(), LLM_GRAPH_TYPE_DEFAULT, mctx.get(), status);
    if (!res || status != GGML_STATUS_SUCCESS) {
        llama_batch_free(batch);
        return false;
    }

    ggml_tensor * logits = res->get_logits();
    if (!logits) {
        llama_batch_free(batch);
        return false;
    }

    if (!ggml_backend_sched_alloc_graph(ctx->get_sched(), res->get_gf())) {
        llama_batch_free(batch);
        return false;
    }

    res->set_inputs(&mctx->get_ubatch());

    status = ctx->graph_compute(res->get_gf(), total_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        llama_batch_free(batch);
        return false;
    }

    const int64_t n_vocab   = logits->ne[0];
    const int64_t n_outputs = logits->ne[1];
    const int64_t n_targets = std::min<int64_t>(n_outputs, response_tokens.size());
    if (n_targets <= 0) {
        llama_batch_free(batch);
        return false;
    }

    std::vector<float> logits_data(static_cast<size_t>(n_vocab * n_outputs));
    ggml_backend_tensor_get(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));

    const float eps = 1e-6f;
    float       nll = 0.0f;
    for (int64_t o = 0; o < n_targets; ++o) {
        const llama_token target = response_tokens[static_cast<size_t>(o)];
        if (target < 0 || target >= n_vocab) {
            continue;
        }

        const float * col = logits_data.data() + o * n_vocab;
        float         max_logit = col[0];
        for (int64_t v = 1; v < n_vocab; ++v) {
            if (col[v] > max_logit) {
                max_logit = col[v];
            }
        }

        float sum_exp = 0.0f;
        for (int64_t v = 0; v < n_vocab; ++v) {
            sum_exp += std::exp(col[v] - max_logit);
        }
        if (sum_exp <= 0.0f) {
            continue;
        }

        const float logp = col[target] - max_logit - std::log(sum_exp + eps);
        nll += -logp;
    }

    nll /= static_cast<float>(n_targets);
    loss_out = nll;

    llama_batch_free(batch);
    return true;
}

bool compute_entropy_loss(llama_context *                  ctx,
                          llama_model *                    model,
                          const std::vector<float> &       prefix,
                          int32_t                          prefix_len,
                          int32_t                          n_embd,
                          const std::vector<llama_token> & prompt_tokens,
                          const std::vector<llama_token> & generated,
                          const std::vector<llama_token> & style_tokens,
                          const std::vector<llama_token> & list_tokens,
                          const std::vector<float> &       style_target,
                          float                            style_weight,
                          float                            list_weight,
                          float                            style_embed_weight,
                          float                            repeat_weight,
                          float &                          loss_out,
                          float &                          entropy_out) {
    if (generated.empty()) {
        return false;
    }

    std::vector<llama_token> full_tokens = prompt_tokens;
    full_tokens.insert(full_tokens.end(), generated.begin(), generated.end());

    std::vector<float> token_embd;
    if (!ttsv::token_embeddings(*model, full_tokens, token_embd)) {
        return false;
    }

    std::vector<float> full_embd;
    full_embd.resize(prefix.size() + token_embd.size());
    std::memcpy(full_embd.data(), prefix.data(), prefix.size() * sizeof(float));
    std::memcpy(full_embd.data() + prefix.size(), token_embd.data(), token_embd.size() * sizeof(float));

    const int32_t       total_tokens = prefix_len + static_cast<int32_t>(full_tokens.size());
    std::vector<int8_t> logits_mask(total_tokens, 0);
    const int32_t       gen_start = prefix_len + static_cast<int32_t>(prompt_tokens.size());
    for (int32_t i = gen_start; i < total_tokens; ++i) {
        logits_mask[i] = 1;
    }

    llama_memory_clear(ctx->get_memory(), true);

    llama_batch batch = llama_batch_init(total_tokens, n_embd, 1);
    fill_batch_embd(batch, full_embd, n_embd, 0, logits_mask);

    llama_batch_allocr balloc(model->hparams.n_pos_per_embd());
    if (!balloc.init(batch, model->vocab, ctx->get_memory(), n_embd, ctx->n_seq_max(), false)) {
        llama_batch_free(batch);
        return false;
    }

    auto mctx = ctx->get_memory()->init_batch(balloc, ctx->n_ubatch(), false);
    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        llama_batch_free(batch);
        return false;
    }

    ggml_status status = GGML_STATUS_SUCCESS;
    auto *      res    = ctx->build_graph_for_batch(mctx->get_ubatch(), LLM_GRAPH_TYPE_DEFAULT, mctx.get(), status);
    if (!res || status != GGML_STATUS_SUCCESS) {
        llama_batch_free(batch);
        return false;
    }

    ggml_tensor * logits = res->get_logits();
    if (!logits) {
        llama_batch_free(batch);
        return false;
    }

    const int64_t n_vocab   = logits->ne[0];
    const int64_t n_outputs = logits->ne[1];
    ggml_tensor * loss      = build_entropy_loss(res->get_ctx(), logits, n_outputs);

    ggml_set_output(loss);
    ggml_build_forward_expand(res->get_gf(), loss);

    if (!ggml_backend_sched_alloc_graph(ctx->get_sched(), res->get_gf())) {
        llama_batch_free(batch);
        return false;
    }

    res->set_inputs(&mctx->get_ubatch());

    status = ctx->graph_compute(res->get_gf(), total_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        llama_batch_free(batch);
        return false;
    }

    float loss_val = 0.0f;
    ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(loss_val));
    entropy_out = loss_val;

    if ((style_weight > 0.0f && !style_tokens.empty()) || (list_weight > 0.0f && !list_tokens.empty())) {
        std::vector<float> logits_data(static_cast<size_t>(n_vocab * n_outputs));
        ggml_backend_tensor_get(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));

        auto in_vocab = [n_vocab](llama_token tok) {
            return tok >= 0 && tok < n_vocab;
        };

        std::vector<llama_token> style_in_vocab;
        std::vector<llama_token> list_in_vocab;
        style_in_vocab.reserve(style_tokens.size());
        list_in_vocab.reserve(list_tokens.size());
        for (auto tok : style_tokens) {
            if (in_vocab(tok)) {
                style_in_vocab.push_back(tok);
            }
        }
        for (auto tok : list_tokens) {
            if (in_vocab(tok)) {
                list_in_vocab.push_back(tok);
            }
        }

        float       style_loss = 0.0f;
        float       list_loss  = 0.0f;
        const float eps        = 1e-6f;

        for (int64_t o = 0; o < n_outputs; ++o) {
            const float * col       = logits_data.data() + o * n_vocab;
            float         max_logit = col[0];
            for (int64_t v = 1; v < n_vocab; ++v) {
                if (col[v] > max_logit) {
                    max_logit = col[v];
                }
            }

            float sum_exp = 0.0f;
            for (int64_t v = 0; v < n_vocab; ++v) {
                sum_exp += std::exp(col[v] - max_logit);
            }
            if (sum_exp <= 0.0f) {
                continue;
            }

            if (!style_in_vocab.empty() && style_weight > 0.0f) {
                float style_exp = 0.0f;
                for (auto tok : style_in_vocab) {
                    style_exp += std::exp(col[tok] - max_logit);
                }
                const float p_style = style_exp / sum_exp;
                style_loss += -std::log(p_style + eps);
            }

            if (!list_in_vocab.empty() && list_weight > 0.0f) {
                float list_exp = 0.0f;
                for (auto tok : list_in_vocab) {
                    list_exp += std::exp(col[tok] - max_logit);
                }
                const float p_list = list_exp / sum_exp;
                list_loss += p_list;
            }
        }

        if (n_outputs > 0) {
            if (!style_in_vocab.empty() && style_weight > 0.0f) {
                style_loss /= float(n_outputs);
                loss_val += style_weight * style_loss;
            }
            if (!list_in_vocab.empty() && list_weight > 0.0f) {
                list_loss /= float(n_outputs);
                loss_val += list_weight * list_loss;
            }
        }
    }

    if (style_embed_weight > 0.0f && !style_target.empty() && style_target.size() == static_cast<size_t>(n_embd) &&
        !generated.empty()) {
        std::vector<float> gen_mean(n_embd, 0.0f);
        const size_t       offset = prompt_tokens.size();
        for (size_t i = 0; i < generated.size(); ++i) {
            const float * row = token_embd.data() + (offset + i) * n_embd;
            for (int32_t d = 0; d < n_embd; ++d) {
                gen_mean[d] += row[d];
            }
        }
        const float inv = 1.0f / static_cast<float>(generated.size());
        for (int32_t d = 0; d < n_embd; ++d) {
            gen_mean[d] *= inv;
        }

        float dot    = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (int32_t d = 0; d < n_embd; ++d) {
            dot += gen_mean[d] * style_target[d];
            norm_a += gen_mean[d] * gen_mean[d];
            norm_b += style_target[d] * style_target[d];
        }
        const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom > 0.0f) {
            const float cos_sim          = dot / denom;
            const float style_embed_loss = 1.0f - cos_sim;
            loss_val += style_embed_weight * style_embed_loss;
        }
    }

    if (repeat_weight > 0.0f && generated.size() > 1) {
        std::unordered_set<llama_token> uniq;
        uniq.reserve(generated.size());
        int32_t repeat_adj = 0;
        for (size_t i = 0; i < generated.size(); ++i) {
            uniq.insert(generated[i]);
            if (i > 0 && generated[i] == generated[i - 1]) {
                repeat_adj += 1;
            }
        }
        const float repeat_ratio     = 1.0f - (float) uniq.size() / (float) generated.size();
        const float repeat_adj_ratio = (float) repeat_adj / (float) (generated.size() - 1);
        const float repeat_loss      = 0.5f * repeat_ratio + 0.5f * repeat_adj_ratio;
        loss_val += repeat_weight * repeat_loss;
    }

    loss_out = loss_val;

    llama_batch_free(batch);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTSV)) {
        return 1;
    }

    if (params.n_gpu_layers != 0) {
        fprintf(stderr, "llama-ttsv-train: forcing CPU only (n_gpu_layers=0)\n");
        params.n_gpu_layers = 0;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    if (!llama_init) {
        fprintf(stderr, "llama-ttsv-train: failed to load model\n");
        return 1;
    }

    llama_context * ctx   = llama_init->context();
    llama_model *   model = llama_init->model();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "llama-ttsv-train: invalid model/context\n");
        return 1;
    }

    common_chat_templates_ptr chat_templates;
    if (params.enable_chat_template) {
        chat_templates = common_chat_templates_init(model, params.chat_template);
    }

    const int32_t n_embd     = llama_model_n_embd_inp(model);
    const int32_t prefix_len = params.ttsv_prefix_length;
    const bool    data_init  = should_data_init(*model);

    if (prefix_len <= 0) {
        fprintf(stderr, "llama-ttsv-train: prefix length must be > 0\n");
        return 1;
    }

    auto prompts = ttsv::load_prompts(params.ttsv_prompts_file);
    if (prompts.empty()) {
        fprintf(stderr, "llama-ttsv-train: no prompts loaded from %s\n", params.ttsv_prompts_file.c_str());
        return 1;
    }

    const std::vector<std::string> style_pieces = {
        "I",      " I",     "I'm",     " I'm",     "me",   " me",   "my",    " my",    "you",    " you",
        "your",   " your",  "we",      " we",      "us",   " us",   "you are", " you are",
        "i am",   " i am",  "thank you", " thank you",
        "with you", " with you", "together", " together", "here", " here", "safe", " safe",
        "gentle", " gentle", "soft", " soft", "breathe", " breathe", "breath", " breath",
        "stay", " stay", "close", " close", "calm", " calm", "rest", " rest", "listen", " listen",
    };
    const std::vector<std::string> list_pieces = {
        "\n", "\n\n", "\n-", " -", "-", "\n1", "1.", "2.", "3.", "\n1.", "\n2.", "\n3.", "*", " *", ":", " :",
    };
    const auto                     style_tokens = build_token_set(ctx, style_pieces);
    const auto                     list_tokens  = build_token_set(ctx, list_pieces);
    const std::vector<std::string> style_refs   = {
        "I am here with you. You are safe. I care about you. We will get through this together.",
        "I hear you. I am with you tonight. You are not alone.",
        "I am proud of you. You can breathe. I will stay with you.",
        "Welcome, traveler. The fire is warm. You are safe here with me tonight.",
    };
    std::vector<float> style_target;
    if (!compute_mean_embedding(ctx, model, style_refs, style_target)) {
        style_target.clear();
    }

    std::vector<style_pair_tokens> style_pairs;
    if (params.ttsv_style_nll_weight > 0.0f && !params.ttsv_style_pairs_file.empty()) {
        style_pairs = load_style_pairs(ctx, chat_templates.get(), params, params.ttsv_style_pairs_file);
        if (style_pairs.empty()) {
            fprintf(stderr, "llama-ttsv-train: no style pairs loaded from %s\n", params.ttsv_style_pairs_file.c_str());
        }
    }

    float base_lr     = params.ttsv_lr;
    float base_lr_min = params.ttsv_lr_min;
    if (data_init && std::fabs(params.ttsv_lr - 1e-3f) < 1e-9f && std::fabs(params.ttsv_lr_min - 1e-5f) < 1e-9f) {
        base_lr     = 5e-6f;
        base_lr_min = 1e-6f;
        fprintf(stderr, "llama-ttsv-train: using Llama-style lr defaults (lr=%.6g lr_min=%.6g)\n", base_lr,
                base_lr_min);
    }

    std::vector<float> prefix;
    int32_t            loaded_tokens = 0;
    int32_t            loaded_embd   = 0;
    if (!params.ttsv_path.empty()) {
        if (!ttsv::load_prefix(params.ttsv_path, loaded_tokens, loaded_embd, prefix)) {
            fprintf(stderr, "llama-ttsv-train: failed to load prefix from %s\n", params.ttsv_path.c_str());
            return 1;
        }
        if (loaded_tokens != prefix_len || loaded_embd != n_embd) {
            fprintf(stderr, "llama-ttsv-train: prefix shape mismatch (file %dx%d, expected %dx%d)\n", loaded_tokens,
                    loaded_embd, prefix_len, n_embd);
            return 1;
        }
    } else {
        prefix.resize(static_cast<size_t>(prefix_len) * static_cast<size_t>(n_embd));
        if (data_init) {
            std::vector<float> mean;
            std::vector<float> stdev;
            int64_t            token_count = 0;
            if (compute_embedding_stats(ctx, model, chat_templates.get(), params, prompts, mean, stdev, token_count)) {
                init_prefix_data_driven(prefix, n_embd, mean, stdev, params.ttsv_seed);
                fprintf(stderr, "llama-ttsv-train: data-driven init from %lld tokens\n",
                        static_cast<long long>(token_count));
            } else {
                fprintf(stderr, "llama-ttsv-train: failed data-driven init, using random\n");
                init_prefix_random(prefix, params.ttsv_seed);
            }
        } else {
            init_prefix_random(prefix, params.ttsv_seed);
        }
    }

    adamw_state opt_state;
    opt_state.m.assign(prefix.size(), 0.0f);
    opt_state.v.assign(prefix.size(), 0.0f);

    const int32_t max_new_tokens = params.n_predict > 0 ? params.n_predict : 128;
    const float   base_perturb   = params.ttsv_perturb;
    std::mt19937  rng(params.ttsv_seed);

    const int64_t     total_steps = static_cast<int64_t>(params.ttsv_epochs) * static_cast<int64_t>(prompts.size());
    std::deque<float> entropy_window;
    int32_t           collapse_hits = 0;
    float             lr_scale      = 1.0f;
    float             perturb_scale = 1.0f;

    int64_t step = 0;
    for (int32_t epoch = 0; epoch < params.ttsv_epochs; ++epoch) {
        for (const auto & prompt : prompts) {
            step += 1;

            const std::string      formatted     = format_prompt(chat_templates.get(), params, prompt, true);
            const auto             prompt_tokens = common_tokenize(ctx, formatted, true, true);
            common_params_sampling sampling      = params.sampling;
            if (params.ttsv_list_logit_bias != 0.0f && !list_tokens.empty()) {
                const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                for (auto tok : list_tokens) {
                    if (tok >= 0 && tok < n_vocab) {
                        sampling.logit_bias.push_back({tok, params.ttsv_list_logit_bias});
                    }
                }
            }
            sampling.seed                        = static_cast<uint32_t>(params.ttsv_seed + step);
            common_sampler_ptr sampler(common_sampler_init(model, sampling));
            auto               generated =
                generate_response(ctx, model, sampler.get(), prefix, prefix_len, n_embd, prompt_tokens, max_new_tokens);

            if (generated.empty()) {
                fprintf(stderr, "llama-ttsv-train: empty generation for prompt, skipping\n");
                continue;
            }

            int style_pair_idx = -1;
            if (!style_pairs.empty() && params.ttsv_style_nll_weight > 0.0f) {
                std::uniform_int_distribution<size_t> pair_dist(0, style_pairs.size() - 1);
                style_pair_idx = static_cast<int>(pair_dist(rng));
            }

            float base_loss = 0.0f;
            float entropy_loss = 0.0f;
            if (!compute_entropy_loss(ctx, model, prefix, prefix_len, n_embd, prompt_tokens, generated, style_tokens,
                                      list_tokens, style_target, params.ttsv_style_weight, params.ttsv_list_weight,
                                      params.ttsv_style_embed_weight, params.ttsv_repeat_weight, base_loss,
                                      entropy_loss)) {
                fprintf(stderr, "llama-ttsv-train: failed to compute base loss, skipping\n");
                continue;
            }

            float style_nll = 0.0f;
            if (style_pair_idx >= 0) {
                const auto & pair = style_pairs[static_cast<size_t>(style_pair_idx)];
                if (compute_supervised_nll_loss(ctx, model, prefix, prefix_len, n_embd, pair.prompt_tokens,
                                                pair.response_tokens, style_nll)) {
                    base_loss += params.ttsv_style_nll_weight * style_nll;
                }
            }

            if (params.ttsv_collapse_window > 0 && params.ttsv_collapse_patience > 0) {
                entropy_window.push_back(entropy_loss);
                if (entropy_window.size() > static_cast<size_t>(params.ttsv_collapse_window)) {
                    entropy_window.pop_front();
                }
                if (entropy_window.size() == static_cast<size_t>(params.ttsv_collapse_window)) {
                    float avg = 0.0f;
                    for (float v : entropy_window) {
                        avg += v;
                    }
                    avg /= static_cast<float>(entropy_window.size());
                    if (avg < params.ttsv_entropy_floor) {
                        collapse_hits += 1;
                    } else {
                        collapse_hits = 0;
                    }
                    if (collapse_hits >= params.ttsv_collapse_patience) {
                        lr_scale      = std::max(lr_scale * 0.5f, 1e-3f);
                        perturb_scale = std::max(perturb_scale * 0.5f, 0.05f);
                        collapse_hits = 0;
                        entropy_window.clear();
                        fprintf(stderr,
                                "llama-ttsv-train: entropy collapse detected (avg %.6f < %.6f), lr_scale %.3g "
                                "perturb_scale %.3g\n",
                                avg, params.ttsv_entropy_floor, lr_scale, perturb_scale);
                    }
                }
            }

            const float                        perturb = std::max(1e-8f, base_perturb * perturb_scale);
            std::vector<float>                 delta(prefix.size());
            std::uniform_int_distribution<int> sign_dist(0, 1);
            for (auto & v : delta) {
                v = sign_dist(rng) ? 1.0f : -1.0f;
            }

            std::vector<float> prefix_plus  = prefix;
            std::vector<float> prefix_minus = prefix;
            for (size_t i = 0; i < prefix.size(); ++i) {
                const float d = perturb * delta[i];
                prefix_plus[i] += d;
                prefix_minus[i] -= d;
            }

            float loss_plus  = 0.0f;
            float loss_minus = 0.0f;
            float entropy_dummy = 0.0f;
            if (!compute_entropy_loss(ctx, model, prefix_plus, prefix_len, n_embd, prompt_tokens, generated,
                                      style_tokens, list_tokens, style_target, params.ttsv_style_weight,
                                      params.ttsv_list_weight, params.ttsv_style_embed_weight,
                                      params.ttsv_repeat_weight, loss_plus, entropy_dummy) ||
                !compute_entropy_loss(ctx, model, prefix_minus, prefix_len, n_embd, prompt_tokens, generated,
                                      style_tokens, list_tokens, style_target, params.ttsv_style_weight,
                                      params.ttsv_list_weight, params.ttsv_style_embed_weight,
                                      params.ttsv_repeat_weight, loss_minus, entropy_dummy)) {
                fprintf(stderr, "llama-ttsv-train: failed to compute SPSA loss, skipping\n");
                continue;
            }

            if (style_pair_idx >= 0) {
                const auto & pair = style_pairs[static_cast<size_t>(style_pair_idx)];
                float style_nll_plus = 0.0f;
                if (compute_supervised_nll_loss(ctx, model, prefix_plus, prefix_len, n_embd, pair.prompt_tokens,
                                                pair.response_tokens, style_nll_plus)) {
                    loss_plus += params.ttsv_style_nll_weight * style_nll_plus;
                }
                float style_nll_minus = 0.0f;
                if (compute_supervised_nll_loss(ctx, model, prefix_minus, prefix_len, n_embd, pair.prompt_tokens,
                                                pair.response_tokens, style_nll_minus)) {
                    loss_minus += params.ttsv_style_nll_weight * style_nll_minus;
                }
            }

            const float        grad_scale = (loss_plus - loss_minus) / (2.0f * perturb);
            std::vector<float> grad(prefix.size());
            for (size_t i = 0; i < grad.size(); ++i) {
                grad[i] = grad_scale * delta[i];
            }

            const float lr = linear_lr(base_lr, base_lr_min, step, total_steps) * lr_scale;
            adamw_step(prefix, grad, opt_state, lr, 0.9f, 0.999f, params.ttsv_eps, params.ttsv_weight_decay);

            if (step % 1 == 0) {
                fprintf(stderr,
                        "epoch %d/%d step %lld/%lld entropy %.6f total_loss %.6f style_nll %.6f loss+ %.6f loss- %.6f lr %.6g\n",
                        epoch + 1, params.ttsv_epochs, static_cast<long long>(step),
                        static_cast<long long>(total_steps), entropy_loss, base_loss, style_nll, loss_plus, loss_minus,
                        lr);
            }
        }

        if (!ttsv::save_prefix(params.ttsv_out, prefix_len, n_embd, prefix)) {
            fprintf(stderr, "llama-ttsv-train: failed to save prefix to %s\n", params.ttsv_out.c_str());
            return 1;
        }
    }

    if (!ttsv::save_prefix(params.ttsv_out, prefix_len, n_embd, prefix)) {
        fprintf(stderr, "llama-ttsv-train: failed to save prefix to %s\n", params.ttsv_out.c_str());
        return 1;
    }

    llama_backend_free();

    return 0;
}
