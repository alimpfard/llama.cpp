#include "models.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache.h"
#include <cstdlib>

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP (Qwen3.5/3.6): extra decoder block appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);

    // KVA projector metadata (optional)
    ml.get_key(LLM_KV_KVA_SPLIT,     hparams.kva_split,     false);
    ml.get_key(LLM_KV_KVA_EMBD,      hparams.kva_embd,      false);
    ml.get_key(LLM_KV_KVA_BLOCKS,    hparams.kva_blocks,    false);
    ml.get_key(LLM_KV_KVA_HEADS,     hparams.kva_heads,     false);
    ml.get_key(LLM_KV_KVA_FFN,       hparams.kva_ffn,       false);
    ml.get_key(LLM_KV_KVA_ROPE_BASE, hparams.kva_rope_base, false);
    if (hparams.kva_split > 0) {
        GGML_ASSERT(hparams.kva_split < hparams.n_layer() && hparams.kva_embd > 0 && hparams.kva_heads > 0);
    }
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_impl");

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }

    // KVA projector: trunk placed with the last exact layer, per-layer heads with their layer
    if (hparams.kva_split > 0 && !mtp_only) {
        const int     ks = (int) hparams.kva_split - 1;
        const int64_t d  = hparams.kva_embd;
        const int64_t ff = hparams.kva_ffn;
        const int     kf = TENSOR_NOT_REQUIRED;
        kva_inp_norm = create_tensor(tn(LLM_TENSOR_KVA_INP_NORM, "weight", ks), { n_embd }, kf);
        kva_inp      = create_tensor(tn(LLM_TENSOR_KVA_INP,      "weight", ks), { n_embd, d }, kf);
        kva_out_norm = create_tensor(tn(LLM_TENSOR_KVA_OUT_NORM, "weight", ks), { d }, kf);
        kva_blocks.resize(hparams.kva_blocks);
        for (uint32_t i = 0; i < hparams.kva_blocks; ++i) {
            auto & b = kva_blocks[i];
            // NOTE: block index doubles as the placement layer (kva.blk.%d); fine for layer splits where
            //       the first few layers share a device with layer kva_split-1
            b.attn_norm = create_tensor(tn(LLM_TENSOR_KVA_BLK_ATTN_NORM, "weight", i), { d }, kf);
            b.attn_qkv  = create_tensor(tn(LLM_TENSOR_KVA_BLK_ATTN_QKV,  "weight", i), { d, 3 * d }, kf);
            b.attn_out  = create_tensor(tn(LLM_TENSOR_KVA_BLK_ATTN_OUT,  "weight", i), { d, d }, kf);
            b.ffn_norm  = create_tensor(tn(LLM_TENSOR_KVA_BLK_FFN_NORM,  "weight", i), { d }, kf);
            b.ffn_up    = create_tensor(tn(LLM_TENSOR_KVA_BLK_FFN_UP,    "weight", i), { d, ff }, kf);
            b.ffn_gate  = create_tensor(tn(LLM_TENSOR_KVA_BLK_FFN_GATE,  "weight", i), { d, ff }, kf);
            b.ffn_down  = create_tensor(tn(LLM_TENSOR_KVA_BLK_FFN_DOWN,  "weight", i), { ff, d }, kf);
        }
        kva_heads.resize(n_layer);
        const int64_t key_dim   = hparams.ssm_d_state * hparams.ssm_n_group;
        const int64_t value_dim = hparams.ssm_d_inner;
        const int64_t n_v_heads = hparams.ssm_dt_rank;
        for (int il = (int) hparams.kva_split; il < n_layer; ++il) {
            const int64_t out = hparams.is_recr(il) ? (2 * key_dim + value_dim + 2 * n_v_heads) : (2 * n_embd_k_gqa);
            kva_heads[il].w = create_tensor(tn(LLM_TENSOR_KVA_HEAD, "weight", il), { d, out }, kf);
            kva_heads[il].b = create_tensor(tn(LLM_TENSOR_KVA_HEAD, "bias",   il), { out }, kf);
        }
        // optional heads: residual inputs of selected late layers and the final normed hidden state (drafter features)
        kva_res.resize(n_layer);
        for (int il = (int) hparams.kva_split; il < n_layer; ++il) {
            kva_res[il].w = create_tensor(tn(LLM_TENSOR_KVA_RES, "weight", il), { d, n_embd }, kf | TENSOR_NOT_REQUIRED);
            kva_res[il].b = create_tensor(tn(LLM_TENSOR_KVA_RES, "bias",   il), { n_embd },    kf | TENSOR_NOT_REQUIRED);
        }
        kva_final.w = create_tensor(tn(LLM_TENSOR_KVA_FINAL, "weight"), { d, n_embd }, kf | TENSOR_NOT_REQUIRED);
        kva_final.b = create_tensor(tn(LLM_TENSOR_KVA_FINAL, "bias"),   { n_embd },    kf | TENSOR_NOT_REQUIRED);
        {
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();

    const auto & qm = static_cast<const llama_model_qwen35 &>(model);
    const bool kva = kva_active(qm);
    const int  n_layer_exact = kva ? (int) hparams.kva_split : n_layer;
    // causal [n_tokens, n_tokens] mask for the projector's own attention (f16 when flash-attn is on)
    auto * inp_kva_mask = kva && !getenv("KVA_NOFA") ? build_attn_inp_no_cache() : nullptr;  // unused inputs may not exist

    // with KVA the graph output is the last token only, so the out_ids input would be unused (and unallocated)
    ggml_tensor * inp_out_ids = (kva && n_outputs == 1) ? nullptr : build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer_exact; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    if (kva) {
        res->kva_used = true;
        // approximate the late layers' cache state for all tokens but the last, run the last exactly
        cur = build_kva(qm, inp, inp_kva_mask, inpL, inp_pos, sections);
    } else {
        cur = inpL;
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    if (kva && cparams.embeddings_nextn && !cparams.embeddings_nextn_masked) {
        // MTP drafting wants the final hidden state of every token; the approximated tokens get the
        // normed layer-`split` input as a stand-in (decoding stays exact, the drafter loses some acceptance)
        ggml_tensor * h_ap;
        if (qm.kva_final.w) {                                                      // predicted final normed hidden state
            h_ap = ggml_add(ctx0, build_lora_mm(qm.kva_final.w, kva_trunk), qm.kva_final.b);
            cb(h_ap, "kva_final", -1);
        } else {
            h_ap = ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens - 1, inpL->nb[1], 0));
            h_ap = build_norm(h_ap, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        }
        ggml_tensor * h_all = ggml_concat(ctx0, h_ap, cur, 1);                    // [n_embd, n_tokens]
        ggml_set_output(h_all);
        ggml_build_forward_expand(gf, h_all);
        res->t_h_nextn = h_all;
    } else {
        res->t_h_nextn = cur;
    }

    if (kva) {
        // cur already holds only the last token; the output ids (if any) point at it
        if (inp_out_ids && n_outputs == 0) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
    } else if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il,
                    int64_t   n_seq_tokens) {
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il,
        int64_t              n_seq_tokens_override) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = n_seq_tokens_override > 0 ? n_seq_tokens_override : ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_seq_tokens_override > 0 || ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il, n_seq_tokens);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}


// ============================================================================================
// KVA: late-layer KV approximation for prefill.
//
// The first kva_split layers run exactly. A small projector maps the residual after layer
// kva_split-1 to every late layer's token-mixer inputs (deltanet: q,k,v pre-conv + a + b;
// attention: k,v pre-norm) and only the cheap state updates run for those tokens (conv+silu,
// delta-net scan, k_norm+rope, KV cache write). The last token of the ubatch goes through the late
// layers exactly, so its logits (and any DFlash taps) are what the exact model would produce
// given the approximated context. Decode is untouched.
// ============================================================================================

bool llama_model_qwen35::graph::kva_active(const llama_model_qwen35 & qm) const {
    if (!cparams.kva_prefill || !cparams.kva_ubatch || !qm.has_kva() || hparams.kva_split == 0) {
        return false;
    }
    if (n_tokens < 2 || ubatch.n_seqs != 1) {
        return false;
    }
    // v1: outputs may only be requested for the last token of the ubatch
    if (ubatch.output) {
        for (uint32_t i = 0; i + 1 < n_tokens; ++i) {
            if (ubatch.output[i]) {
                return false;
            }
        }
    }
    return true;
}

ggml_tensor * llama_model_qwen35::graph::build_kva(
        const llama_model_qwen35 & qm,
        llm_graph_input_mem_hybrid * inp,
        llm_graph_input_attn_no_cache * inp_mask,
        ggml_tensor * h,
        ggml_tensor * inp_pos,
        int *         sections) {
    const int     split = (int) hparams.kva_split;
    const int64_t n_ap  = n_tokens - 1;
    const int64_t d     = hparams.kva_embd;
    const int64_t nh    = hparams.kva_heads;
    const int64_t hd    = d / nh;

    ggml_tensor * h_ap   = ggml_view_2d(ctx0, h, n_embd, n_ap, h->nb[1], 0);
    ggml_tensor * h_tail = ggml_cont(ctx0, ggml_view_2d(ctx0, h, n_embd, 1, h->nb[1], n_ap * h->nb[1]));
    // positions: M-RoPE stores n_pos components component-major ([t...][h...][w...][0...]);
    // the projector uses the plain first component, the model's rope_multi wants all components
    const int64_t n_pos = inp_pos->ne[0] / n_tokens;
    const size_t  pes   = ggml_element_size(inp_pos);
    ggml_tensor * pos_ap   = ggml_view_1d(ctx0, inp_pos, n_ap, 0);
    ggml_tensor * pos_ap_m = ggml_reshape_1d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, inp_pos, n_ap, n_pos, n_tokens * pes, 0)), n_ap * n_pos);
    ggml_tensor * pos_tail = ggml_reshape_1d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, inp_pos, 1, n_pos, n_tokens * pes, n_ap * pes)), n_pos);
    cb(h_ap, "kva_h_ap", split);

    // ---- projector trunk on the approximated tokens
    ggml_tensor * x = build_norm(h_ap, qm.kva_inp_norm, nullptr, LLM_NORM_RMS, split);
    x = build_lora_mm(qm.kva_inp, x);
    cb(x, "kva_inp", split);
    for (size_t bi = 0; bi < qm.kva_blocks.size(); ++bi) {
        const auto & b = qm.kva_blocks[bi];
        ggml_tensor * xn  = build_norm(x, b.attn_norm, nullptr, LLM_NORM_RMS, split);
        ggml_tensor * qkv = build_lora_mm(b.attn_qkv, xn);                       // [3d, n_ap]
        const size_t es = ggml_element_size(qkv);
        ggml_tensor * q = ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, hd, nh, n_ap, hd * es, qkv->nb[1], 0));
        ggml_tensor * k = ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, hd, nh, n_ap, hd * es, qkv->nb[1], d * es));
        ggml_tensor * v = ggml_cont(ctx0, ggml_view_3d(ctx0, qkv, hd, nh, n_ap, hd * es, qkv->nb[1], 2 * d * es));
        q = ggml_rope_ext(ctx0, q, pos_ap, nullptr, (int) hd, GGML_ROPE_TYPE_NORMAL, 0, hparams.kva_rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos_ap, nullptr, (int) hd, GGML_ROPE_TYPE_NORMAL, 0, hparams.kva_rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * o;
        if (getenv("KVA_NOFA")) {
            // reference path: explicit causal softmax attention
            q = ggml_permute(ctx0, q, 0, 2, 1, 3);                                    // [hd, n_ap, nh]
            k = ggml_permute(ctx0, k, 0, 2, 1, 3);
            ggml_tensor * vt = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));    // [n_ap, hd, nh]
            ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);                              // [n_ap(k), n_ap(q), nh]
            kq = ggml_diag_mask_inf(ctx0, kq, 0);
            kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f / sqrtf((float) hd), 0.0f);
            o = ggml_mul_mat(ctx0, vt, kq);                                           // [hd, n_ap(q), nh]
            o = ggml_cont_2d(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3), d, n_ap);       // [d, n_ap]
        } else {
            // the ubatch causal mask restricted to the approximated tokens (flash-attn wants it contiguous)
            ggml_tensor * m = inp_mask->get_kq_mask();                                // [n_tokens, n_tokens]
            m = ggml_cont(ctx0, ggml_view_2d(ctx0, m, n_ap, n_ap, m->nb[1], 0));
            o = build_attn_mha(q, k, v, nullptr, m, nullptr, nullptr, 1.0f / sqrtf((float) hd), split);  // [d, n_ap]
        }
        x = ggml_add(ctx0, x, build_lora_mm(b.attn_out, o));
        ggml_tensor * xf = build_norm(x, b.ffn_norm, nullptr, LLM_NORM_RMS, split);
        xf = build_ffn(xf,
                b.ffn_up,   nullptr, nullptr,
                b.ffn_gate, nullptr, nullptr,
                b.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, split);
        x = ggml_add(ctx0, x, xf);
        cb(x, "kva_blk", (int) bi);
    }
    x = build_norm(x, qm.kva_out_norm, nullptr, LLM_NORM_RMS, split);
    cb(x, "kva_trunk", split);
    kva_trunk = x;

    // ---- proposed mixer inputs -> cache state for the approximated tokens
    std::vector<kva_rs> rs(n_layer);
    const char * dbg = getenv("KVA_DBG");
    const int kva_dbg = dbg ? atoi(dbg) : 0;
    for (int il = split; il < n_layer && kva_dbg != 1; ++il) {
        ggml_tensor * o = build_lora_mm(qm.kva_heads[il].w, x);
        o = ggml_add(ctx0, o, qm.kva_heads[il].b);                                // [out_il, n_ap]
        cb(o, "kva_head", il);
        if (hparams.is_recr(il)) {
            rs[il] = build_kva_linear(inp->get_recr(), o, il, n_ap);
        } else {
            build_kva_attn(inp->get_attn(), o, pos_ap_m, sections, il, n_ap);
        }
    }

    // ---- the last token, exactly, on top of the approximated state
    ggml_tensor * cur = h_tail;
    for (int il = split; il < n_layer; ++il) {
        // late-layer inputs are exact for the tail token only; consumers of per-token layer inputs
        // (e.g. a DFlash2 drafter) get the layer-`split` input as a stand-in for the approximated tokens
        const bool want_inp = il < (int) cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il];
        if (want_inp) {
            ggml_tensor * ap = ggml_cont(ctx0, h_ap);                                  // default stand-in: layer-split input
            if (qm.kva_res[il].w) {                                                    // predicted residual input of layer il
                ap = ggml_add(ctx0, build_lora_mm(qm.kva_res[il].w, kva_trunk), qm.kva_res[il].b);
                cb(ap, "kva_res", il);
            }
            ggml_tensor * t = ggml_concat(ctx0, ap, cur, 1);                          // [n_embd, n_tokens]
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
            res->t_layer_inp[il] = t;
        } else {
            res->t_layer_inp[il] = cur;
        }
        ggml_tensor * inpSA = cur;
        ggml_tensor * hn = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * a;
        if (hparams.is_recr(il)) {
            a = kva_dbg == 1 ? build_layer_attn_linear(inp->get_recr(), hn, il, /*n_seq_tokens=*/1)
                             : build_layer_attn_linear_tail(inp->get_recr(), hn, il, rs[il]);
        } else {
            a = build_layer_attn_tail(inp->get_attn(), hn, pos_tail, sections, il, n_ap);
        }
        cur = ggml_add(ctx0, a, inpSA);
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * pn = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_layer_ffn(pn, il);
        cur = ggml_add(ctx0, cur, ffn_residual);
        cur = build_cvec(cur, il);
        cb(cur, "kva_tail_out", il);
    }
    return cur;
}

llama_model_qwen35::graph::kva_rs llama_model_qwen35::graph::build_kva_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        o,
        int                  il,
        int64_t              n_ap) {
    const auto * mctx_cur = inp->mctx;
    const int64_t d_inner     = hparams.ssm_d_inner;
    const int64_t n_seqs      = 1;
    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = d_inner / num_v_heads;
    const int64_t key_dim     = head_k_dim * num_k_heads;
    const int64_t value_dim   = head_v_dim * num_v_heads;
    const int64_t qkv_dim     = 2 * key_dim + value_dim;
    const size_t  es          = ggml_element_size(o);

    ggml_tensor * qkv_mixed = ggml_cont(ctx0, ggml_view_2d(ctx0, o, qkv_dim, n_ap, o->nb[1], 0));
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_dim, n_ap, n_seqs);
    ggml_tensor * alpha = ggml_cont(ctx0, ggml_view_2d(ctx0, o, num_v_heads, n_ap, o->nb[1], qkv_dim * es));
    ggml_tensor * braw  = ggml_cont(ctx0, ggml_view_2d(ctx0, o, num_v_heads, n_ap, o->nb[1], (qkv_dim + num_v_heads) * es));

    ggml_tensor * beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, braw, 1, num_v_heads, n_ap, n_seqs));
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_ap, n_seqs);
    ggml_tensor * gate = ggml_mul(ctx0, ggml_softplus(ctx0, ggml_add(ctx0, alpha, model.layers[il].ssm_dt)), model.layers[il].ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_ap, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    // conv state group 0 (after the last token) is written by the exact tail step; with recurrent-state
    // rollback the groups j = 1..K-1 (state after token n_tokens-1-j = approximated token n_ap-j) come from here
    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il, /*write_states=*/false);
    if (cparams.n_rs_seq > 0) {
        const auto    kv_head   = mctx_cur->get_head();
        const auto    mem_size  = mctx_cur->get_size();
        const int64_t row_count = (conv_kernel_size - 1) * conv_channels;
        const size_t  row_size  = ggml_row_size(conv_states_all->type, row_count);
        for (int64_t j = 1; j <= (int64_t) cparams.n_rs_seq && j <= n_ap; ++j) {
            const int64_t s_idx = n_ap - j + 1;                                // rows [s_idx, s_idx + k - 2]
            ggml_tensor * snap = ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, s_idx));
            ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all, row_count, n_seqs, conv_states_all->nb[1],
                    ((size_t) j * mem_size + kv_head) * row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, snap, dst));
        }
    }
    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_ap, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_ap, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_ap, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_ap, key_dim * ggml_element_size(conv_out));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_ap, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_ap, 2 * key_dim * ggml_element_size(conv_out));

    const float eps_norm = hparams.f_norm_rms_eps;
    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    // run the scan; the state is handed to the exact tail step in-graph (re-reading the cache would
    // re-zero a sequence that starts in this ubatch), which then writes the final states
    kva_rs r;
    if (cparams.n_rs_seq == 0) {
        auto dn = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il);
        ggml_build_forward_expand(gf, dn.first);
        r.state = dn.second;                                                   // [S_v, S_v, H_v, n_seqs]
    } else {
        // recurrent-state rollback (speculative decoding): the cache keeps K = n_rs_seq + 1 snapshot groups,
        // group i = state after token (n_tokens - 1 - i). The exact tail step writes group 0; the states
        // after the last approximated tokens fill groups 1..K-1 (see build_recurrent_attn for the layout)
        const int64_t K = cparams.n_rs_seq + 1;
        ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q_conv, k_conv, v_conv, gate, beta, state, K);
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, gdn_out, il});
        ggml_build_forward_expand(gf, gdn_out);
        const int64_t D               = head_v_dim * head_v_dim * num_v_heads;
        const int64_t attn_score_elems = head_v_dim * num_v_heads * n_ap * n_seqs;
        const int64_t snap_elems       = D * n_seqs;
        const size_t  ts               = ggml_element_size(gdn_out);
        r.state = ggml_view_4d(ctx0, gdn_out, head_v_dim, head_v_dim, num_v_heads, n_seqs,
                head_v_dim * ts, head_v_dim * head_v_dim * ts, D * ts, attn_score_elems * ts);   // snapshot 0 = final state
        const int64_t n_snap = std::min<int64_t>(n_ap, K - 1);                 // snapshots 0..n_snap-1 -> groups 1..n_snap
        const auto   kv_head  = mctx_cur->get_head();
        const auto   mem_size = mctx_cur->get_size();
        const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);
        // staged through a contiguous copy: copying straight out of the strided view while the tail scan
        // also reads gdn_out produced corrupted states (buffer reuse in the allocator, most likely)
        ggml_tensor * src = ggml_cont(ctx0, ggml_view_3d(ctx0, gdn_out, D, n_seqs, n_snap,
                D * ts, snap_elems * ts, attn_score_elems * ts));
        ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all, D, n_seqs, n_snap,
                ssm_states_all->nb[1], (size_t) mem_size * row_size,
                (size_t) kv_head * row_size + (size_t) mem_size * row_size);   // start at group 1
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
    }
    const int64_t n_prev = conv_kernel_size - 1;
    r.conv_prev = ggml_view_3d(ctx0, conv_input, n_prev, conv_channels, n_seqs,
            conv_input->nb[1], conv_input->nb[2],
            ggml_row_size(conv_input->type, conv_input->ne[0] - n_prev));      // last k-1 time rows
    cb(r.state, "kva_state", il);
    return r;
}

// exact deltanet layer for the single tail token, continuing from the in-graph approximated state
ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear_tail(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il,
        const kva_rs &       rs) {
    const auto * mctx_cur = inp->mctx;
    const auto   kv_head  = mctx_cur->get_head();
    const auto   mem_size = mctx_cur->get_size();

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = 1;
    const int64_t n_seq_tokens = 1;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t key_dim      = head_k_dim * num_k_heads;
    const int64_t value_dim    = head_v_dim * num_v_heads;
    const int64_t qkv_dim      = 2 * key_dim + value_dim;

    auto qkvz = build_qkvz(cur, il, n_seq_tokens);
    ggml_tensor * qkv_mixed = qkvz.first;                                     // [qkv_dim, 1, 1]
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs));
    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * gate = ggml_mul(ctx0, ggml_softplus(ctx0, ggml_add(ctx0, alpha, model.layers[il].ssm_dt)), model.layers[il].ssm_a);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    // conv context = last k-1 approximated tokens (in-graph), then this token
    ggml_tensor * conv_input = ggml_concat(ctx0, rs.conv_prev, ggml_transpose(ctx0, qkv_mixed), 0);
    {
        const int64_t row_count = (conv_kernel_size - 1) * conv_channels;
        const size_t  row_size  = ggml_row_size(conv_states_all->type, row_count);
        ggml_tensor * conv_state_last = ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, conv_input->ne[0] - (conv_kernel_size - 1)));
        ggml_tensor * conv_state_update = ggml_view_2d(ctx0, conv_states_all, row_count, n_seqs, conv_states_all->nb[1], kv_head * row_size);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_state_last, conv_state_update));
        GGML_UNUSED(mem_size);
    }
    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    const int64_t nb1_qkv = ggml_row_size(conv_out->type, qkv_dim);
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, key_dim * ggml_element_size(conv_out));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 2 * key_dim * ggml_element_size(conv_out));

    const float eps_norm = hparams.f_norm_rms_eps;
    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    // scan from the approximated state and write the final state to the cache
    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, rs.state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
    cb(cur, "kva_tail_linear_out", il);
    return cur;
}

void llama_model_qwen35::graph::build_kva_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             o,
        ggml_tensor *             pos_ap,
        int *                     sections,
        int                       il,
        int64_t                   n_ap) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t kdim = n_embd_head * n_head_kv;
    const size_t  es   = ggml_element_size(o);

    ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, o, kdim, n_ap, o->nb[1], 0));
    ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, o, kdim, n_ap, o->nb[1], kdim * es));
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_ap);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    Kcur = ggml_rope_multi(ctx0, Kcur, pos_ap, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_ap);
    // quantized KV caches store Hadamard-rotated K/V (see build_attn); the rows written here must match
    if (inp->self_k_rot) { Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp->self_k_rot); }
    if (inp->self_v_rot) { Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp->self_v_rot); }
    cb(Kcur, "kva_Kcur", il);
    cb(Vcur, "kva_Vcur", il);

    const auto * mctx_cur = inp->mctx;
    if (!kva_k_idxs_ap) {
        kva_k_idxs_ap = ggml_view_1d(ctx0, inp->get_k_idxs(), n_ap, 0);
        kva_v_idxs_ap = ggml_view_1d(ctx0, inp->get_v_idxs(), n_ap, 0);
    }
    ggml_tensor * k_idxs = kva_k_idxs_ap;
    ggml_tensor * v_idxs = kva_v_idxs_ap;
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, v_idxs, il));
}

// exact attention layer for the single tail token: its own K/V go into the cache, then it attends
// over the whole cache (which now holds the approximated K/V of the earlier tokens)
ggml_tensor * llama_model_qwen35::graph::build_layer_attn_tail(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             pos_tail,
        int *                     sections,
        int                       il,
        int64_t                   i_tail) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t nt = 1;

    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, nt,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, nt,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, nt);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, nt);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, nt);
    Qcur = ggml_rope_multi(ctx0, Qcur, pos_tail, nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, pos_tail, nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    const auto * mctx_cur = inp->mctx;
    ggml_tensor * k_idxs_all = inp->get_k_idxs();
    ggml_tensor * v_idxs_all = inp->get_v_idxs();
    if (!kva_k_idxs_tail) {
        kva_k_idxs_tail = ggml_view_1d(ctx0, k_idxs_all, 1, i_tail * ggml_element_size(k_idxs_all));
        kva_v_idxs_tail = ggml_view_1d(ctx0, v_idxs_all, 1, i_tail * ggml_element_size(v_idxs_all));
    }
    ggml_tensor * k_idxs = kva_k_idxs_tail;
    ggml_tensor * v_idxs = kva_v_idxs_tail;
    if (inp->self_k_rot) {
        Qcur = llama_mul_mat_hadamard(ctx0, Qcur, inp->self_k_rot);
        Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp->self_k_rot);
    }
    if (inp->self_v_rot) { Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp->self_v_rot); }
    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, Vcur);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, v_idxs, il));

    ggml_tensor * kq_mask = inp->get_kq_mask();                                   // [n_kv, n_tokens(pad)]
    ggml_tensor * mask_tail = ggml_view_2d(ctx0, kq_mask, kq_mask->ne[0], nt, kq_mask->nb[1], i_tail * kq_mask->nb[1]);
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    ggml_tensor * out = build_attn_mha(Qcur, k, v, nullptr, mask_tail, nullptr, nullptr, kq_scale, il);
    if (inp->self_v_rot) { out = llama_mul_mat_hadamard(ctx0, out, inp->self_v_rot); }
    cb(out, "kva_tail_attn", il);
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, gate));
    out = build_lora_mm(model.layers[il].wo, out, model.layers[il].wo_s);
    return out;
}
