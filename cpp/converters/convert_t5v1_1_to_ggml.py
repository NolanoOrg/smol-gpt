import io
import os
import sys
import struct
import json
import code
import torch, torch.nn as nn
import numpy as np

import transformers as tf

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

if len(sys.argv) < 3:
    print("Usage: python convert-hf-to-ggml.py model_name dir-output [use-f32]")
    print("  model_name: name of the model to convert. Example: 'bigscience/bloomz-560m'")
    print("  dir-output: directory where the output file will be written")
    print("  use-f32:    if present, use float32 instead of float16")
    sys.exit(1)

model_name = sys.argv[1]
dir_out = sys.argv[2]

# make sure the output directory exists
os.makedirs(dir_out, exist_ok=True)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]
ftype = 1
if len(sys.argv) > 3:
    ftype = 0

print("Loading tokenizer: ", model_name)
tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
print("Loading model: ", model_name)
if "google/" in model_name:
    model = tf.T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16 if ftype == 1 else torch.float32) # pylint: disable=no-member
elif "Salesforce/blip2" in model_name:
    model = tf.Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16 if ftype == 1 else torch.float32) # pylint: disable=no-member
    with torch.no_grad():
        def save_as_fp16(model, path):
            """Save model as fp16"""
            import copy
            model_copy = copy.deepcopy(model)
            if isinstance(model_copy, dict):
                for k, v in model_copy.items():
                    model_copy[k] = v.half()
                    model_copy[k].requires_grad = False
            else:
                model_copy = model_copy.half()
                model_copy.requires_grad = False
            torch.save(model_copy, path)
        # Save model.vision_model at fp16
        print("Saving vision model at fp16")
        save_as_fp16(model.vision_model.state_dict(), dir_out + "/vision-model.bin")
        # Save model.query_tokens
        print("Saving query tokens")
        save_as_fp16(model.query_tokens, dir_out + "/query-tokens.bin")
        # Save model.qformer
        print("Saving qformer")
        save_as_fp16(model.qformer.state_dict(), dir_out + "/bert.bin")
        # Save model.language_projection
        print("Saving language projection")
        save_as_fp16(model.language_projection, dir_out + "/t5_project.bin")
        # Save encoder embedding
        print("Saving encoder embedding")
        save_as_fp16(model.language_model.encoder.embed_tokens, dir_out + "/t5_embed.bin")
        model = model.language_model
else:
    raise Exception("Unknown model name: ", model_name)
model.eval()
for p in model.parameters():
    p.requires_grad = False
hparams = model.config.to_dict()
print("Model loaded: ", model_name)

fname_out = dir_out + f"/ggml-model-{model_name.split('/')[-1]}-{ftype_str[ftype]}.bin"
fout = open(fname_out, "wb")

hparams["multiple_of"] = 1
fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", 256)) # max_seq_length
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["d_ff"]))
fout.write(struct.pack("i", hparams["num_heads"]))
fout.write(struct.pack("i", hparams["num_layers"]))
fout.write(struct.pack("i", hparams["num_decoder_layers"]))
fout.write(struct.pack("i", hparams["relative_attention_num_buckets"]))
fout.write(struct.pack("i", ftype))

# Is this correct??
dot_token = tokenizer.encode(".")[0]
for i in range(hparams["vocab_size"]):
    text = tokenizer.decode([i]).encode('utf-8')
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

# raise Exception("Uncomment the next line")
# import transformers as tf, torch, torch.nn as nn
# model = tf.T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
list_vars = model.state_dict()
with torch.no_grad():
    class Catcher(nn.Module):
        def __init__(self, module, destination_list):
            super().__init__()
            self.module = module
            self.destination_list = destination_list
        def forward(self,
                    hidden_states,
                    mask=None,
                    key_value_states=None,
                    position_bias=None,
                    past_key_value=None,
                    layer_head_mask=None,
                    query_length=None,
                    use_cache=False,
                    output_attentions=False,
                    **kwargs):
            op = self.module(
                hidden_states=hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_value=past_key_value,
                layer_head_mask=layer_head_mask,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions)
            self.destination_list.append(op[2])
            return op
    new_list_vars = []
    model.encoder.block[0].layer[0].SelfAttention = Catcher(
        model.encoder.block[0].layer[0].SelfAttention, new_list_vars)
    model.decoder.block[0].layer[0].SelfAttention = Catcher(
        model.decoder.block[0].layer[0].SelfAttention, new_list_vars)
    fake_input = torch.LongTensor([[1]*256]) # pylint: disable=no-member
    print("Starting forward pass to get relation attention bias...")
    _ = model.generate(fake_input, min_length=257, max_length=257)
    del list_vars['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
    del list_vars['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
    list_vars['encoder.relative_attention_bias'] = new_list_vars[0].squeeze()
    # Pad last dimension of each new_list_vars[1:] to 256
    for i in range(1, len(new_list_vars)):
        new_list_vars[i] = torch.nn.functional.pad(new_list_vars[i], (0, 256-new_list_vars[i].shape[-1]), 'constant', 0)
    list_vars['decoder.relative_attention_bias'] = torch.cat(new_list_vars[1:], dim=-2).squeeze() # pylint: disable=no-member
    assert list_vars['encoder.relative_attention_bias'].shape == (hparams["num_heads"], 256, 256)
    assert list_vars['decoder.relative_attention_bias'].shape == (hparams["num_heads"], 256, 256)
    # if True:
    #     torch.save({"decoder.relative_attention_bias": list_vars['decoder.relative_attention_bias'], "encoder.relative_attention_bias": list_vars['encoder.relative_attention_bias']}, "relative_attention_bias.pt")
del list_vars['decoder.embed_tokens.weight']
del list_vars['encoder.embed_tokens.weight']

for name in list_vars.keys():
    # No gradients for these
    list_vars[name].requires_grad = False
    src = name
    nn = name

    print(src, ' -> ', name)
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float32)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    # default type is fp32
    ftype_cur = 0
    if ftype == 1 and n_dims == 2:
        print("  Converting to float16", data.shape, data[:3, :3].tolist())
        data = data.astype(np.float16)
        ftype_cur = 1
    else:
        print("  Converting to float32", data.shape, data.flatten().tolist()[:3] if n_dims > 1 else data[:3].tolist())
        data = data.astype(np.float32)

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    print(str, data.shape, n_dims)
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
