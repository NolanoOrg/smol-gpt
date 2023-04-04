#include "ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

////////////////////////////////////////////////////////
//////////////////////   T5 V1_1    ////////////////////
////////////////////////////////////////////////////////

struct t5v1_1_hparams {
    int32_t n_vocab           = 32128;
    int32_t n_ctx             = 1024;
    int32_t n_embd            = 4096;
    int32_t n_ffd             = 10240;
    int32_t n_head            = 64;
    int32_t n_encoder_layer   = 24;
    int32_t n_decoder_layer   = 24;
    int32_t relative_attn_num_buckets = 32;
    int32_t f16     = 1;
};

struct t5v1_1_encoder_layer {
    // normalization
    struct ggml_tensor * ln_1_g;

    // attention
    struct ggml_tensor * attn_c_attn_k;
    struct ggml_tensor * attn_c_attn_q;
    struct ggml_tensor * attn_c_attn_v;
    struct ggml_tensor * attn_c_attn_out;

    // normalization
    struct ggml_tensor * ln_2_g;

    // mlp
    struct ggml_tensor * mlp_c_fc_w_0;
    struct ggml_tensor * mlp_c_fc_w_1;
    struct ggml_tensor * mlp_c_fc_wout;
};

struct t5v1_1_decoder_layer {
    // normalization
    struct ggml_tensor * ln_1_g;

    // attention
    struct ggml_tensor * attn_c_attn_q;
    struct ggml_tensor * attn_c_attn_k;
    struct ggml_tensor * attn_c_attn_v;
    struct ggml_tensor * attn_c_attn_out;

    // normalization
    struct ggml_tensor * ln_2_g;

    // cross attention
    struct ggml_tensor * cross_attn_c_attn_k;
    struct ggml_tensor * cross_attn_c_attn_q;
    struct ggml_tensor * cross_attn_c_attn_v;
    struct ggml_tensor * cross_attn_c_attn_out;

    // normalization
    struct ggml_tensor * ln_3_g;

    // mlp
    struct ggml_tensor * mlp_c_fc_w_0;
    struct ggml_tensor * mlp_c_fc_w_1;
    struct ggml_tensor * mlp_c_fc_wout;
};

struct t5v1_1_model {
    struct t5v1_1_hparams hparams;
    struct ggml_context * ctx;

    std::map<std::string, struct ggml_tensor *>tensors;

    std::vector<struct t5v1_1_encoder_layer> encoder_layers;
    std::vector<struct t5v1_1_decoder_layer> decoder_layers;

    struct ggml_tensor * wte;
    struct ggml_tensor * encoder_attn_relative_bias; // Size 256 * 256 * n_head
    struct ggml_tensor * decoder_attn_relative_bias; // Size 256 * 256 * n_head

    struct ggml_tensor * encoder_final_layer_norm_weight;
    struct ggml_tensor * decoder_final_layer_norm_weight;

    struct ggml_tensor * lm_head_weight;

    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;
    struct ggml_tensor * memory_k_cross; // TODO: remove the need for this.
    struct ggml_tensor * memory_v_cross; // TODO: remove the need for this.

    struct ggml_tensor * embd_encoder_hidden;
};

bool t5v1_1_model_load(const std::string & fname, t5v1_1_model & model, gpt_vocab & vocab, int n_ctx) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        printf("%s: failed to open file '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }

        else{
            printf("%s: valid model file '%s' (good magic)\n", __func__, fname.c_str());
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *)&hparams.n_vocab,                   sizeof(hparams.n_vocab));
        fin.read((char *)&hparams.n_ctx,                     sizeof(hparams.n_ctx));
        fin.read((char *)&hparams.n_embd,                    sizeof(hparams.n_embd));
        fin.read((char *)&hparams.n_ffd,                     sizeof(hparams.n_ffd));
        fin.read((char *)&hparams.n_head,                    sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_encoder_layer,           sizeof(hparams.n_encoder_layer));
        fin.read((char *)&hparams.n_decoder_layer,           sizeof(hparams.n_decoder_layer));
        fin.read((char *)&hparams.relative_attn_num_buckets, sizeof(hparams.relative_attn_num_buckets));
        fin.read((char *)&hparams.f16,                       sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;

        printf("%s: hparams: n_vocab = %d\n",                   __func__, hparams.n_vocab);
        printf("%s: hparams: n_ctx   = %d\n",                   __func__, hparams.n_ctx);
        printf("%s: hparams: n_embd  = %d\n",                   __func__, hparams.n_embd);
        printf("%s: hparams: n_ffd   = %d\n",                   __func__, hparams.n_ffd);
        printf("%s: hparams: n_head  = %d\n",                   __func__, hparams.n_head);
        printf("%s: hparams: n_encoder_layer = %d\n",           __func__, hparams.n_encoder_layer);
        printf("%s: hparams: n_decoder_layer = %d\n",           __func__, hparams.n_decoder_layer);
        printf("%s: hparams: relative_attn_num_buckets = %d\n", __func__, hparams.relative_attn_num_buckets);
        printf("%s: hparams: f16    = %d\n",                    __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        n_vocab = model.hparams.n_vocab;
        // fin.read((char *)&n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16) {
        case 0: wtype = GGML_TYPE_F32;  break;
        case 1: wtype = GGML_TYPE_F16;  break;
        case 2: wtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = GGML_TYPE_Q4_1; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            __func__, fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    const ggml_type wtype2 = GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_vocab                   = hparams.n_vocab;
        const int n_ctx                     = hparams.n_ctx;
        const int n_embd                    = hparams.n_embd;
        const int n_head                    = hparams.n_head;
        const int n_ffd                     = hparams.n_ffd;
        const int n_encoder_layer           = hparams.n_encoder_layer;
        const int n_decoder_layer           = hparams.n_decoder_layer;
        const int relative_attn_num_buckets = hparams.relative_attn_num_buckets;

        {
            // non-encoder and non-decoder
            ctx_size += n_vocab * n_embd * ggml_type_sizef(wtype); // wte
            // ctx_size += relative_attn_num_buckets * n_head * ggml_type_sizef(GGML_TYPE_F32); // attn_relative_bias
            ctx_size += n_head * 256 * 256 * ggml_type_sizef(GGML_TYPE_F32); // encoder_attn_relative_bias
            ctx_size += n_head * 256 * 256 * ggml_type_sizef(GGML_TYPE_F32); // decoder_attn_relative_bias

            ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // encoder_final_layer_norm_weight
            ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // decoder_final_layer_norm_weight

            ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype); // lm_head_weight
        }

        {
            // encoder
            ctx_size += n_encoder_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g

            ctx_size += n_encoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_k
            ctx_size += n_encoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_q
            ctx_size += n_encoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_v
            ctx_size += n_encoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_out

            ctx_size += n_encoder_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g

            ctx_size += n_encoder_layer * (n_embd * n_ffd * ggml_type_sizef(wtype)); // mlp_c_fc_w_0
            ctx_size += n_encoder_layer * (n_embd * n_ffd * ggml_type_sizef(wtype)); // mlp_c_fc_w_1
            ctx_size += n_encoder_layer * (n_embd * n_ffd * ggml_type_sizef(wtype)); // mlp_c_fc_wout
        }

        {
            // decoder
            ctx_size += n_decoder_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g

            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_k
            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_q
            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_v
            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // attn_c_attn_out

            ctx_size += n_decoder_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_2_g

            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // cross_attn_c_attn_k
            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // cross_attn_c_attn_q
            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // cross_attn_c_attn_v
            ctx_size += n_decoder_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // cross_attn_c_attn_out

            ctx_size += n_decoder_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_3_g

            ctx_size += n_decoder_layer * (n_embd * n_ffd * ggml_type_sizef(wtype)); // mlp_c_fc_w_0
            ctx_size += n_decoder_layer * (n_embd * n_ffd * ggml_type_sizef(wtype)); // mlp_c_fc_w_1
            ctx_size += n_decoder_layer * (n_embd * n_ffd * ggml_type_sizef(wtype)); // mlp_c_fc_wout
        }

        ctx_size += n_ctx * n_decoder_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_k TODO: check
        ctx_size += n_ctx * n_decoder_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_v TODO: check
        ctx_size += n_ctx * n_decoder_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_k_cross TODO: check
        ctx_size += n_ctx * n_decoder_layer * n_embd * ggml_type_sizef(GGML_TYPE_F32); // memory_v_cross TODO: check
        ctx_size += 512 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // embd_encoder_hidden  TODO: check

        ctx_size += (6 + (9 * n_encoder_layer) + (14 * n_decoder_layer)) * 256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));

    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }

        else {
            printf("ggml context created successfully\n");
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_vocab                   = hparams.n_vocab;
        const int n_ctx                     = hparams.n_ctx;
        const int n_embd                    = hparams.n_embd;
        const int n_head                    = hparams.n_head;
        const int n_ffd                     = hparams.n_ffd;
        const int n_encoder_layer           = hparams.n_encoder_layer;
        const int n_decoder_layer           = hparams.n_decoder_layer;
        const int relative_attn_num_buckets = hparams.relative_attn_num_buckets;

        model.encoder_layers.resize(n_encoder_layer);
        model.decoder_layers.resize(n_decoder_layer);

        model.wte                        = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.encoder_attn_relative_bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 256, n_head);
        model.decoder_attn_relative_bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 256, n_head);

        model.encoder_final_layer_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.decoder_final_layer_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.lm_head_weight = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        model.tensors["shared.weight"] = model.wte;
        model.tensors["encoder.relative_attention_bias"] = model.encoder_attn_relative_bias;
        model.tensors["decoder.relative_attention_bias"] = model.decoder_attn_relative_bias;

        model.tensors["encoder.final_layer_norm.weight"] = model.encoder_final_layer_norm_weight;
        model.tensors["decoder.final_layer_norm.weight"] = model.decoder_final_layer_norm_weight;

        model.tensors["lm_head.weight"] = model.lm_head_weight;

        for (int i = 0; i < n_encoder_layer; i++) {
            auto & layer = model.encoder_layers[i];

            layer.ln_1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.attn_c_attn_k   = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.attn_c_attn_q   = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.attn_c_attn_v   = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.attn_c_attn_out = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ln_2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.mlp_c_fc_w_0  = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ffd);
            layer.mlp_c_fc_w_1  = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ffd);
            layer.mlp_c_fc_wout = ggml_new_tensor_2d(ctx, wtype, n_ffd, n_embd);

            // map by name
            model.tensors["encoder.block." + std::to_string(i) + ".layer.0.layer_norm.weight"] = layer.ln_1_g;

            model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.k.weight"] = layer.attn_c_attn_k;
            model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.q.weight"] = layer.attn_c_attn_q;
            model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.v.weight"] = layer.attn_c_attn_v;
            model.tensors["encoder.block." + std::to_string(i) + ".layer.0.SelfAttention.o.weight"] = layer.attn_c_attn_out;

            model.tensors["encoder.block." + std::to_string(i) + ".layer.1.layer_norm.weight"] = layer.ln_2_g;

            model.tensors["encoder.block." + std::to_string(i) + ".layer.1.DenseReluDense.wi_0.weight"] = layer.mlp_c_fc_w_0;
            model.tensors["encoder.block." + std::to_string(i) + ".layer.1.DenseReluDense.wi_1.weight"] = layer.mlp_c_fc_w_1;
            model.tensors["encoder.block." + std::to_string(i) + ".layer.1.DenseReluDense.wo.weight"] = layer.mlp_c_fc_wout;
        }
        for (int i = 0; i < n_decoder_layer; i++) {
            auto & layer = model.decoder_layers[i];

            layer.ln_1_g                = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.attn_c_attn_k         = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.attn_c_attn_q         = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.attn_c_attn_v         = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.attn_c_attn_out       = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ln_2_g                = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.cross_attn_c_attn_k   = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.cross_attn_c_attn_q   = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.cross_attn_c_attn_v   = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.cross_attn_c_attn_out = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.ln_3_g                = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.mlp_c_fc_w_0          = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ffd);
            layer.mlp_c_fc_w_1          = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ffd);
            layer.mlp_c_fc_wout         = ggml_new_tensor_2d(ctx, wtype, n_ffd, n_embd);

            // map by name
            model.tensors["decoder.block." + std::to_string(i) + ".layer.0.layer_norm.weight"] = layer.ln_1_g;

            model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.k.weight"] = layer.attn_c_attn_k;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.q.weight"] = layer.attn_c_attn_q;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.v.weight"] = layer.attn_c_attn_v;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.0.SelfAttention.o.weight"] = layer.attn_c_attn_out;

            model.tensors["decoder.block." + std::to_string(i) + ".layer.1.layer_norm.weight"] = layer.ln_2_g;

            model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.k.weight"] = layer.cross_attn_c_attn_k;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.q.weight"] = layer.cross_attn_c_attn_q;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.v.weight"] = layer.cross_attn_c_attn_v;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.1.EncDecAttention.o.weight"] = layer.cross_attn_c_attn_out;

            model.tensors["decoder.block." + std::to_string(i) + ".layer.2.layer_norm.weight"] = layer.ln_3_g;

            model.tensors["decoder.block." + std::to_string(i) + ".layer.2.DenseReluDense.wi_0.weight"] = layer.mlp_c_fc_w_0;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.2.DenseReluDense.wi_1.weight"] = layer.mlp_c_fc_w_1;
            model.tensors["decoder.block." + std::to_string(i) + ".layer.2.DenseReluDense.wo.weight"] = layer.mlp_c_fc_wout;
        }
    }


    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_decoder_layer = hparams.n_decoder_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_decoder_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        model.memory_k            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_k_cross      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v_cross      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = { 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                // print the actual value and in the file
                printf("Actual dim = %d\n", ggml_nelements(tensor));
                printf("In file dim = %d\n", nelements);
                printf("ne[0] = %d, ne[1] = %d\n, ne[2] = %d", ne[0], ne[1], ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            if (0) {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype) {
                case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                default:
                        {
                            fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                            return false;
                        }
            };

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }


    fin.close();

    return true;
}

bool t5v1_1_compute_encoder_hidden(
        const t5v1_1_model & model,
        const int n_threads,
        const std::vector<float> & embd_encoder_inp,
        const int N, // Number of encoder tokens (N <= 256)
              size_t                     & mem_per_token) {
    const int n_past = 0;
    const auto & hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;
    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_ffd   = hparams.n_ffd;
    const int n_head  = hparams.n_head;
    const int n_encoder_layer = hparams.n_encoder_layer;

    assert(N < 250); // For now, we only support up to 256 tokens for input
    assert(N == embd_encoder_inp.size()/n_embd);

    const int relative_attn_num_buckets   = hparams.relative_attn_num_buckets;

    const int d_key = n_embd/n_head;

    static size_t buf_size = 256u * 1024 * 1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token * N > buf_size) {
        const size_t buf_size_new = 1.5 * (mem_per_token * N); // add 10% to account for ggml object overhead
        printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads };

    // Prepare input embeddings of size [N, n_embd]
    struct ggml_tensor * embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N);
    memcpy(embd->data, embd_encoder_inp.data(), N*n_embd*ggml_element_size(embd));

    // printf("First 5 elements of embd: %f %f %f %f %f\n", ((float *)embd->data)[0], ((float *)embd->data)[1], ((float *)embd->data)[2], ((float *)embd->data)[3], ((float *)embd->data)[4]);
    // printf("Last 5 elements of embd's first row: %f %f %f %f %f\n", ((float *)embd->data)[n_embd-5], ((float *)embd->data)[n_embd-4], ((float *)embd->data)[n_embd-3], ((float *)embd->data)[n_embd-2], ((float *)embd->data)[n_embd-1]);
    // printf("Last 5 elements of emdb = %f %f %f %f %f\n", ((float *)embd->data)[N*n_embd-5], ((float *)embd->data)[N*n_embd-4], ((float *)embd->data)[N*n_embd-3], ((float *)embd->data)[N*n_embd-2], ((float *)embd->data)[N*n_embd-1]);
    // Prepare additive encoder positional bias of size [n_head, N, N] by slicing from model.encoder_attn_relative_bias
    struct ggml_tensor * additive_pos_bias_slice           = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, N, N, n_head);
    // struct ggml_tensor * additive_pos_bias_slice_transpose = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, N, N, n_head);
    {
        int size_per_head_src = 256 * 256;
        int size_per_head_dst = N * N;

        int size_per_row_src = 256;
        int size_per_row_dst = N;

        float * ptr_to_data_src = (float *)model.encoder_attn_relative_bias->data;
        float * ptr_to_data_dst = (float *)additive_pos_bias_slice->data;

        for (int head_idx = 0; head_idx < n_head; ++head_idx) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    float * ptr_to_data_src_ij = ptr_to_data_src + (head_idx * size_per_head_src) + (i * size_per_row_src) + j;
                    float * ptr_to_data_dst_ij = ptr_to_data_dst + (head_idx * size_per_head_dst) + (i * size_per_row_dst) + j;
                    // Copy data from src to dst
                    *ptr_to_data_dst_ij = *ptr_to_data_src_ij;
                }
            }
        }
        // // Transpose
        // ptr_to_data_dst = (float *)additive_pos_bias_slice_transpose->data;
        // for (int head_idx = 0; head_idx < n_head; ++head_idx) {
        //     for (int i = 0; i < N; ++i) {
        //         for (int j = 0; j < N; ++j) {
        //             float * ptr_to_data_src_ij = ptr_to_data_src + (head_idx * size_per_head_src) + (i * size_per_row_src) + j;
        //             float * ptr_to_data_dst_ij = ptr_to_data_dst + (head_idx * size_per_head_dst) + (j * size_per_row_dst) + i;
        //             // Copy data from src to dst
        //             *ptr_to_data_dst_ij = *ptr_to_data_src_ij;
        //         }
        //     }
        // }
        // // additive_pos_bias_slice = additive_pos_bias_slice_transpose;
        // // Print the first 5 and last 5 elements of additive_pos_bias_slice
        // printf("First 5 elements of additive_pos_bias_slice: %f %f %f %f %f\n", ((float *)additive_pos_bias_slice->data)[0], ((float *)additive_pos_bias_slice->data)[1], ((float *)additive_pos_bias_slice->data)[2], ((float *)additive_pos_bias_slice->data)[3], ((float *)additive_pos_bias_slice->data)[4]);
        // printf("Last 5 elements of additive_pos_bias_slice = %f %f %f %f %f\n", ((float *)additive_pos_bias_slice->data)[N*N*n_head-5], ((float *)additive_pos_bias_slice->data)[N*N*n_head-4], ((float *)additive_pos_bias_slice->data)[N*N*n_head-3], ((float *)additive_pos_bias_slice->data)[N*N*n_head-2], ((float *)additive_pos_bias_slice->data)[N*N*n_head-1]);
    }

    // Since we receive embeddings after first layer as input, we don't need embedding table lookup.
    // // wte
    // struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte, embd);
    struct ggml_tensor * inpL = embd;
    for (int il = 0; il < n_encoder_layer; ++il) {
        struct ggml_tensor * cur;
        // norm before attention
        {
            cur = ggml_t5_norm(ctx0, inpL);

            // cur = ln_1_g * cur
            cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.encoder_layers[il].ln_1_g, cur), cur);
        }

        // self-attention
        {
            // Weight
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.encoder_layers[il].attn_c_attn_q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.encoder_layers[il].attn_c_attn_k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.encoder_layers[il].attn_c_attn_v, cur);

            // store key and value to memory
            if (true) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N * n_embd, (ggml_element_size(model.memory_k) * n_embd) * (il * n_ctx + 0));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N * n_embd, (ggml_element_size(model.memory_v) * n_embd) * (il * n_ctx + 0));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0, Qcur, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, //Kcur, //
                                                model.memory_k,
                                                 (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // // No mask needed for encoder.
            // // // KQ_masked = mask_past(KQ_scaled)
            // // struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // Add additive positional bias
            struct ggml_tensor * KQ_biased = ggml_add(ctx0, KQ, additive_pos_bias_slice);
            // struct ggml_tensor * KQ_biased = ggml_add(ctx0, KQ, additive_pos_bias_slice);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_biased);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, //Vcur, //
                                        model.memory_v,
                                                 (n_past + N) * n_embd, il * n_ctx * ggml_element_size(model.memory_v) * n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (first weight)
            cur = ggml_mul_mat(ctx0, model.encoder_layers[il].attn_c_attn_out, cur);
        }

        // add residual
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpL);

        // pre feed-forward layer norm
        {
            cur = ggml_t5_norm(ctx0, inpFF);

            // cur = ln_2_g * cur
            cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.encoder_layers[il].ln_2_g, cur), cur);
        }

        // feed-forward network
        {
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0, model.encoder_layers[il].mlp_c_fc_w_0, cur);

            cur = ggml_mul_mat(ctx0, model.encoder_layers[il].mlp_c_fc_w_1, cur);

            // Gelu activation
            tmp = ggml_gelu(ctx0, tmp);

            // Gating
            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0, model.encoder_layers[il].mlp_c_fc_wout, cur);
        }

        // inpL = inpFF + inpL
        inpL = ggml_add(ctx0, cur, inpFF);
    }

    // norm
    {
        inpL = ggml_t5_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_mul(ctx0, ggml_repeat(ctx0, model.encoder_final_layer_norm_weight, inpL), inpL);
    }

    // // lm_head
    // {
    //     inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);
    // }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    printf("I am here\n");
    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // printf("First 5 elements of inpL = %f %f %f %f %f\n", ((float *) ggml_get_data(inpL))[0], ((float *) ggml_get_data(inpL))[1], ((float *) ggml_get_data(inpL))[2], ((float *) ggml_get_data(inpL))[3], ((float *) ggml_get_data(inpL))[4]);
    // printf("Last 5 elements of inpL = %f %f %f %f %f\n", ((float *) ggml_get_data(inpL))[n_embd*N-5], ((float *) ggml_get_data(inpL))[n_embd*N-4], ((float *) ggml_get_data(inpL))[n_embd*N-3], ((float *) ggml_get_data(inpL))[n_embd*N-2], ((float *) ggml_get_data(inpL))[n_embd*N-1]);
    // sum of abs value of all elements of inpL
    float sum = 0;
    for (int i = 0; i < n_embd*N; i++) {
        sum += fabs(((float *) ggml_get_data(inpL))[i]);
    }
    // printf("Sum of abs value of all elements of inpL = %f\n", sum);
    // printf("First 5 elements of model.encoder_final_layer_norm_weight = %f %f %f %f %f\n", ((float *) model.encoder_final_layer_norm_weight->data)[0], ((float *) model.encoder_final_layer_norm_weight->data)[1], ((float *) model.encoder_final_layer_norm_weight->data)[2], ((float *) model.encoder_final_layer_norm_weight->data)[3], ((float *) model.encoder_final_layer_norm_weight->data)[4]);
    // printf("First 5 elements of model.decoder_layers[0].ln_1_g = %f %f %f %f %f\n", ((float *) model.decoder_layers[0].ln_1_g->data)[0], ((float *) model.decoder_layers[0].ln_1_g->data)[1], ((float *) model.decoder_layers[0].ln_1_g->data)[2], ((float *) model.decoder_layers[0].ln_1_g->data)[3], ((float *) model.decoder_layers[0].ln_1_g->data)[4]);
    // return result for just the last token
    assert(ggml_are_same_shape(model.embd_encoder_hidden, inpL));
    memcpy(model.embd_encoder_hidden->data, (float *) ggml_get_data(inpL), sizeof(float) * n_embd * N);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

void pretty_print_tensor(struct ggml_tensor * t) {
    assert(t->ne[3] == 1); // only support 3D tensors
    assert(GGML_TYPE_F32 == t->type); // only support float32 (float) tensors
    float * ptr_to_data_src = (float *) t->data;
    // Shape of the tensor
    printf("Shape: %d %d %d\n", t->ne[0], t->ne[1], t->ne[2]);
    for (int i = 0; i < t->ne[2]; ++i) {
        if (i > 2) { printf("\n... <EOT>\n\n"); break; }
        for (int j = 0; j < t->ne[1]; ++j) {
            // only the first 3 and last 3 elements are printed
            if (j < 3 || j > t->ne[1] - 4) {
                    for (int k = 0; k < t->ne[0]; ++k) {
                        if (k < 3 || k > t->ne[0] - 4) {
                            printf("%f ", ptr_to_data_src[i * t->ne[1] * t->ne[0] + j * t->ne[0] + k]);
                        }
                        if (k == t->ne[0] - 4) {
                            printf("... ");
                        }
                    }
                printf("\n");
            }
        }
        printf("\n");
    }
}

bool t5v1_1_eval(
        const t5v1_1_model & model,
        const int n_threads,
        const int n_past,
        // const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              int                        num_encoder_input_tokens,
              size_t                     & mem_per_token) {
    assert(num_encoder_input_tokens > 0); // Num encoder input tokens must be greater than 0
    const int N = embd_inp.size();
    assert(N == 1); // Can only generate one token at a time
    assert(N + n_past < 250); // N + n_past must be less than 256

    const auto & hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;
    const int n_ctx   = hparams.n_ctx;
    const int n_embd  = hparams.n_embd;
    const int n_ffd   = hparams.n_ffd;
    const int n_head  = hparams.n_head;
    const int n_decoder_layer = hparams.n_decoder_layer;
    // printf("n_decoder_layer = %d\n", n_decoder_layer);

    const int d_key      = n_embd/n_head;

    static size_t buf_size = 256u * 1024 * 1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token * N > buf_size) {
        const size_t buf_size_new = 1.1 * (mem_per_token * N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = { .n_threads = n_threads };

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    // wte
    // N = 1; n_past = 2;
    struct ggml_tensor * additive_pos_bias_slice           = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, N+n_past, N, n_head);
    // struct ggml_tensor * additive_pos_bias_slice_transpose = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, N, N, n_head);
    {
        int size_per_head_src = 256 * 256;
        int size_per_head_dst = N * (N + n_past);

        int size_per_row_src = 256;
        int size_per_row_dst = N + n_past;

        float * ptr_to_data_src = (float *)model.decoder_attn_relative_bias->data;
        // // Print the data in the model.decoder_attn_relative_bias->data tensor of shape model.decoder_attn_relative_bias->ne (max 4 dims) where last dim is 1
        // printf("Printing the data in the model.decoder_attn_relative_bias->data tensor of shape model.decoder_attn_relative_bias->ne = %d %d %d %d\n", model.decoder_attn_relative_bias->ne[0], model.decoder_attn_relative_bias->ne[1], model.decoder_attn_relative_bias->ne[2], model.decoder_attn_relative_bias->ne[3]);
        // pretty_print_tensor(model.decoder_attn_relative_bias);

        float * ptr_to_data_dst = (float *)additive_pos_bias_slice->data;

        for (int head_idx = 0; head_idx < n_head; ++head_idx) {
            for (int i = n_past; i < N+n_past; ++i) {
                for (int j = 0; j < N+n_past; ++j) {
                    float * ptr_to_data_src_ij = ptr_to_data_src + (head_idx * size_per_head_src) + (i * size_per_row_src) + j;
                    float * ptr_to_data_dst_ij = ptr_to_data_dst + (head_idx * size_per_head_dst) + ((i - n_past) * size_per_row_dst) + j;
                    // Copy data from src to dst
                    *ptr_to_data_dst_ij = *ptr_to_data_src_ij;
                }
            }
        }
        // // Transpose
        // ptr_to_data_dst = (float *)additive_pos_bias_slice_transpose->data;
        // for (int head_idx = 0; head_idx < n_head; ++head_idx) {
        //     for (int i = 0; i < N; ++i) {
        //         for (int j = 0; j < N; ++j) {
        //             float * ptr_to_data_src_ij = ptr_to_data_src + (head_idx * size_per_head_src) + (i * size_per_row_src) + j;
        //             float * ptr_to_data_dst_ij = ptr_to_data_dst + (head_idx * size_per_head_dst) + (j * size_per_row_dst) + i;
        //             // Copy data from src to dst
        //             *ptr_to_data_dst_ij = *ptr_to_data_src_ij;
        //         }
        //     }
        // }
        // // additive_pos_bias_slice = additive_pos_bias_slice_transpose;
        // // Print the first 5 and last 5 elements of additive_pos_bias_slice

        // pretty_print_tensor(additive_pos_bias_slice);
    }

    // struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte, embd);
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte, embd);
    for (int il = 0; il < n_decoder_layer; ++il) {
        struct ggml_tensor * cur;
        // norm before attention
        {
            cur = ggml_t5_norm(ctx0, inpL);

            // cur = ln_1_g * cur
            cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.decoder_layers[il].ln_1_g, cur), cur);
        }

        // self-attention
        {
            // Weight
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.decoder_layers[il].attn_c_attn_q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.decoder_layers[il].attn_c_attn_k, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.decoder_layers[il].attn_c_attn_v, cur);

            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0, Qcur, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, //Kcur, //
                                                model.memory_k,
                                                 (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // Add additive positional bias
            struct ggml_tensor * KQ_biased = ggml_add(ctx0, KQ, additive_pos_bias_slice);

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_biased, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, //Vcur, //
                                        model.memory_v,
                                                 (n_past + N) * n_embd, il * n_ctx * ggml_element_size(model.memory_v) * n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0, KQV_merged, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (first weight)
            cur = ggml_mul_mat(ctx0, model.decoder_layers[il].attn_c_attn_out, cur);
        }
        
        // add residual
        struct ggml_tensor * inpCrossAtt = ggml_add(ctx0, cur, inpL);
        // norm before cross attention
        {
            cur = ggml_t5_norm(ctx0, inpCrossAtt);

            // cur = ln_1_g * cur
            cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.decoder_layers[il].ln_2_g, cur), cur);
        }

        // cross-attention
        {
            // Weight
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.decoder_layers[il].cross_attn_c_attn_q, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.decoder_layers[il].cross_attn_c_attn_k, model.embd_encoder_hidden);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.decoder_layers[il].cross_attn_c_attn_v, model.embd_encoder_hidden);

            struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k_cross, num_encoder_input_tokens * n_embd,
                                                  (ggml_element_size(model.memory_k_cross) * n_embd) * (il * n_ctx + 0));
            struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v_cross, num_encoder_input_tokens * n_embd,
                                                  (ggml_element_size(model.memory_v_cross) * n_embd) * (il * n_ctx + 0));

            ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
            ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_cpy(ctx0, Qcur, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, 0 + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_k_cross,
                                         (0 + num_encoder_input_tokens) * n_embd, il * n_ctx * ggml_element_size(model.memory_k_cross) * n_embd),
                            n_embd/n_head, n_head, 0 + num_encoder_input_tokens),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // No mask needed for cross attention.
            // No additive positional bias for cross attention.

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ);

            // V_trans = Vmem.view(n_embd/n_head, n_head, 0 + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_v_cross,
                                         (0 + num_encoder_input_tokens) * n_embd, il * n_ctx * ggml_element_size(model.memory_v_cross) * n_embd),
                            n_embd/n_head, n_head, 0 + num_encoder_input_tokens),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0, KQV_merged, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (first weight)
            cur = ggml_mul_mat(ctx0, model.decoder_layers[il].cross_attn_c_attn_out, cur);
        }

        // add residual
        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpCrossAtt);
        // /**/struct ggml_tensor * inpFF = inpCrossAtt;

        // pre feed-forward layer norm
        {
            cur = ggml_t5_norm(ctx0, inpFF);

            // cur = ln_2_g * cur
            cur = ggml_mul(ctx0, ggml_repeat(ctx0, model.decoder_layers[il].ln_3_g, cur), cur);
        }

        // feed-forward network
        {
            struct ggml_tensor * tmp = ggml_mul_mat(ctx0, model.decoder_layers[il].mlp_c_fc_w_0, cur);

            cur = ggml_mul_mat(ctx0, model.decoder_layers[il].mlp_c_fc_w_1, cur);

            // Gelu activation
            tmp = ggml_gelu(ctx0, tmp);

            // Gating
            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0, model.decoder_layers[il].mlp_c_fc_wout, cur);
        }

        // inpL = inpFF + inpL
        inpL = ggml_add(ctx0, cur, inpFF);
    }


    // norm
    {
        inpL = ggml_t5_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_mul(ctx0, ggml_repeat(ctx0, model.decoder_final_layer_norm_weight, inpL), inpL);
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.lm_head_weight, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    // pretty_print_tensor(inpL);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main_t5v1_1(gpt_params params) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();
    int64_t t_load_us = 0;

    std::mt19937 rng(params.seed);

    gpt_vocab vocab;
    t5v1_1_model model;
    printf("\nmodel\n");
    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        const int n_ctx = 256; // TODO: set context from user input ??
        if (!t5v1_1_model_load(params.model, model, vocab, n_ctx)) {  // TODO: set context from user input ??
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    bool test_mode = false;
    std::vector<float> embd_encoder_inp;
    if (!test_mode) {
        // read the prompt from file `params.prompt` using standard file io
        printf("Reading prompt from file '%s'\n", params.prompt.c_str());
        std::string prompt;
        {
            FILE *fp = fopen(params.prompt.c_str(), "r");
            if (!fp) {
                fprintf(stderr, "%s: failed to open prompt file '%s'\n", __func__, params.prompt.c_str());
                return 1;
            }

            char buf[1024];
            while (fgets(buf, sizeof(buf), fp)) {
                prompt += buf;
            }
            fclose(fp);
        }
        // tokenize the prompt
        embd_encoder_inp = ::whitespace_tokenize_float(prompt); //TODO: set bos to true?
    } else {
        // initialize with value across flattened matrix of size (model.hparams.n_embd X 128)
        // First row of flattened matrix is added by 0.001, second bỷ 0.002, ... 128th by 0.128
        // First column of flattened matrix is added by 0.001, second bỷ 0.002, ... n_embd by n_embd/1000
        // embd_encoder_inp.resize(model.hparams.n_embd*128, 0.1);
        embd_encoder_inp.resize(model.hparams.n_embd * 128, 0);
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < model.hparams.n_embd; j++) {
                float i_float = (float) i+1;
                float j_float = (float) j+1;
                embd_encoder_inp[i * model.hparams.n_embd + j] = (i_float + j_float)/1000.0;
            }
        }
    }

    // Get size of encoder input
    const int N = embd_encoder_inp.size() / model.hparams.n_embd;
    printf("N = %d\n", N);
    auto & ctx  = model.ctx;
    model.embd_encoder_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_embd, N);

    // Run encoder to first get mem usage over 5 dummy tokens and then over actual tokens
    size_t mem_per_token = 0;
    std::vector<float> dummy_embd_encoder_inp;
    dummy_embd_encoder_inp.resize(model.hparams.n_embd * 10, 0.1);
    t5v1_1_compute_encoder_hidden(model, params.n_threads, dummy_embd_encoder_inp, 10, mem_per_token);
    printf("mem_per_token encoder dummy = %zu\n", mem_per_token);
    t5v1_1_compute_encoder_hidden(model, params.n_threads, embd_encoder_inp, N, mem_per_token);
    printf("mem_per_token encoder true = %zu\n", mem_per_token);

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = {0}; // TODO: set from user input
    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());
    std::vector<gpt_vocab::id> embd;
    printf("\n");
    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());

    // for (int i = 0; i < (int) embd_inp.size(); i++) {
    //     printf("%6d -> '%s'\n", embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    // }
    printf("\n");
    printf("sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    printf("\n\n");

    float * ptr_to_hidden_output = (float *) model.embd_encoder_hidden->data;
    printf("\n<|BEGIN> ");
    if (params.return_logits) {
        for (int k = 0; k < N; k++) {
            printf("logits: ");
            for (int i = 0; i < model.hparams.n_embd; i++) {
                printf("%f ", ptr_to_hidden_output[k*model.hparams.n_embd + i]);
            }
            if (k == N-1) {
                // printf(" <END|>\n");
                printf(" \n");
            } else {
                printf(" \n");
            }
        }
    }

    // return 0;

    // determine the required inference memory per token:
    mem_per_token = 0;
    t5v1_1_eval(model, params.n_threads, 0, { 1 }, logits, N, mem_per_token);
    // t5v1_1_eval(model, params.n_threads, 1, { 1 }, logits, N, mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!t5v1_1_eval(model, params.n_threads, n_past, embd, logits, N, mem_per_token)) { // update logits
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            if (params.return_logits) {
                printf("logits: ");
                for (int i = 0; i < n_vocab; i++) {
                    // Upto 8 decimal places
                    printf("%.8f ", logits[i]);
                }
                printf("\n");
                // Stdout should flush before returning
                fflush(stdout);
                // return 0;
            }

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = sample_top_p_top_k_repeat_penalty(
                        vocab,
                        logits.data() + (logits.size() - n_vocab),
                        last_n_tokens,
                        repeat_penalty,
                        top_k,
                        top_p,
                        temp,
                        rng);

                // // print
                // printf("\ngenerated token: '%s' (%d)\n", vocab.id_to_token[id].c_str(), id);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                if (params.return_logits) {
                    printf("logits: ");
                    for (int i = 0; i < model.hparams.n_vocab; i++) {
                        // Upto 8 decimal places
                        printf("%.8f ", logits[i]);
                    }
                    printf("\n");
                }
                embd.push_back(embd_inp[k]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            if (!params.return_logits) {
                printf(" %d ", id);
            }
            // printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 1) {
            break;
        }
    }
    printf(" <END|>\n");

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

int main(int argc, char ** argv) {
    gpt_params params; // We say "gpt", but it's actually any LLM
    // params.model = "models/ggml-model-bloomz-7b1-f16-q4_0.bin";
    // params.prompt = "Je vais";

    // loop through argv and print all the arguments, one per line
    for (int i = 0; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }
    // return if params.model or params.prompt are empty
    if (params.model.empty() || params.prompt.empty()) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    if (params.prompt.empty()) {
        return 1;
    }

    // Get the model type from argv[1]
    std::string model_type = argv[1];
    printf("model_type: %s\n", model_type.c_str());

    if (params.return_logits) {
        printf("********************************\n");
        printf("*** return_logits mode ***\n");
        printf("*** setting sampling to greedy ***\n");
        printf("********************************\n");
        // model_type should be either gptj or gptneox or bloom
        // if (model_type != "gptj" && model_type != "gptneox" && model_type != "bloom") {
        //     printf("model_type: %s, should be either gptj or gptneox or bloom\n", model_type.c_str());
        //     assert(false);
        // }
    }

    if (model_type == "t5v1_1") {
        return main_t5v1_1(params);
    } else {
        printf("Unknown model type: %s\n", model_type.c_str());
        return 1;
    }
}
