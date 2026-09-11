// kva-eval: continuation NLL after an exact vs. approximated (KVA) prefill.
//
// For each window of (prompt_len + cont_len) tokens from a text file:
//   1. decode the prompt without logits (this is where --kva applies)
//   2. decode the continuation with logits for every token (exact path, on top of that state)
//   3. accumulate NLL of the continuation tokens and greedy top-1 agreement vs. the reference file
// Run twice (with and without --kva) and compare; optionally dump the per-token greedy tokens.
//
// usage: llama-kva-eval -m model.gguf -f text.txt [--kva] [--prompt-len 1536] [--cont-len 512] [--n-seqs 24]
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 4096;
    int prompt_len = 1536, cont_len = 512, n_seqs = 24;
    std::string dump_path;
    // pull our own args out before common parsing
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--prompt-len" && i + 1 < argc) { prompt_len = atoi(argv[++i]); continue; }
        if (a == "--cont-len"   && i + 1 < argc) { cont_len   = atoi(argv[++i]); continue; }
        if (a == "--n-seqs"     && i + 1 < argc) { n_seqs     = atoi(argv[++i]); continue; }
        if (a == "--dump"       && i + 1 < argc) { dump_path  = argv[++i];       continue; }
        args.push_back(argv[i]);
    }
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("failed to load\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> all = common_tokenize(ctx, params.prompt, false, false);
    const int L = prompt_len + cont_len;
    LOG_INF("text: %zu tokens, windows of %d+%d, up to %d sequences, kva=%s\n", all.size(), prompt_len, cont_len, n_seqs, params.kva ? "on" : "off");

    double nll_sum = 0.0; int64_t n_tok = 0;
    std::vector<llama_token> greedy;
    int done = 0;
    for (size_t off = 0; off + L <= all.size() && done < n_seqs; off += L, ++done) {
        llama_memory_clear(llama_get_memory(ctx), true);
        // 1. prompt, no logits
        llama_batch b = llama_batch_init(params.n_batch, 0, 1);
        // the last prompt token is fed with the continuation (it needs logits); the KVA path covers the rest
        for (int i = 0; i < prompt_len - 1; i += params.n_batch) {
            const int n = std::min(params.n_batch, prompt_len - 1 - i);
            common_batch_clear(b);
            for (int j = 0; j < n; ++j) common_batch_add(b, all[off + i + j], i + j, {0}, false);
            if (llama_decode(ctx, b) != 0) { LOG_ERR("decode failed (prompt)\n"); return 1; }
        }
        // 2. continuation with logits; token at prompt_len-1 predicts the first continuation token
        common_batch_clear(b);
        common_batch_add(b, all[off + prompt_len - 1], prompt_len - 1, {0}, true);
        for (int j = 0; j < cont_len - 1; ++j) common_batch_add(b, all[off + prompt_len + j], prompt_len + j, {0}, true);
        if (llama_decode(ctx, b) != 0) { LOG_ERR("decode failed (cont)\n"); return 1; }
        double seq_nll = 0.0;
        std::vector<float> lp(n_vocab);
        for (int j = 0; j < cont_len; ++j) {
            const float * logits = llama_get_logits_ith(ctx, j);
            float mx = -INFINITY; int am = 0;
            for (int t = 0; t < n_vocab; ++t) { if (logits[t] > mx) { mx = logits[t]; am = t; } }
            double z = 0.0;
            for (int t = 0; t < n_vocab; ++t) z += std::exp((double) logits[t] - mx);
            const llama_token truth = all[off + prompt_len + j];
            seq_nll += -((double) logits[truth] - mx - std::log(z));
            greedy.push_back(am);
        }
        nll_sum += seq_nll; n_tok += cont_len;
        LOG_INF("seq %2d: nll %.4f (running ppl %.4f)\n", done, seq_nll / cont_len, std::exp(nll_sum / n_tok));
        llama_batch_free(b);
    }
    const double nll = nll_sum / std::max<int64_t>(n_tok, 1);
    LOG_INF("RESULT kva=%s seqs=%d tokens=%" PRId64 " nll=%.5f ppl=%.4f\n", params.kva ? "on" : "off", done, n_tok, nll, std::exp(nll));
    if (!dump_path.empty()) {
        std::ofstream f(dump_path, std::ios::binary);
        f.write((const char *) greedy.data(), greedy.size() * sizeof(llama_token));
        LOG_INF("wrote %zu greedy tokens to %s\n", greedy.size(), dump_path.c_str());
    }
    llama_backend_free();
    return 0;
}
