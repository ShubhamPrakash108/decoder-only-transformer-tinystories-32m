# Experiment Observations

This document consolidates the development and experiment notes that were previously split across two `observation.txt` files. It records what failed, why it failed, and how the project changed as a result.

## Current repository state

Some entries below describe historical or planned configurations. At the time of consolidation, `config/config.yml` contains one Transformer block, a block size of 128, dropout of 0.2, and a batch size of 500. The historical values and proposed changes are retained because they explain the experiment results; they should not be read as the current configuration.

## 1. Initial refactor and first training run

The original naive implementation was refactored with Gemini and expanded with comments.

The first one-epoch training run loaded and split the dataset successfully:

```text
Dataset:
  X: torch.Size([32098, 16]), torch.int64
  y: torch.Size([32098, 16]), torch.int64

Splits:
  train:      X [25678, 16], y [25678, 16]
  validation: X [ 3209, 16], y [ 3209, 16]
  test:       X [ 3211, 16], y [ 3211, 16]

DataLoaders:
  train: 803 batches
  validation: 101 batches
  test: 101 batches

Epoch 1/1 | Train Loss: 28.9037 | Val Loss: 16.8910 | LR: 1.00e-05
```

Training completed, but saving failed with a safetensors `RuntimeError` because `final_linear_layer.weight` and `token_embedding.weight` shared the same underlying storage.

### Why it happened

GPT-style models commonly tie the output projection to the input token embedding:

```python
self.final_linear_layer.weight = self.token_embedding.weight
```

This reduces the parameter count and keeps the two mappings consistent. A PyTorch state dict still exposes both names, even though they refer to one tensor. Passing that state dict directly to `safetensors.torch.save_file` is unsafe because safetensors refuses to duplicate aliased memory silently.

The mistaken assumption was that every value returned by `model.state_dict()` was independent. That is usually true for untied models, but not for models with shared parameters.

### Native-checkpoint fix

The native PyTorch training and generation path now uses a two-part contract:

1. Save only the token embedding by removing the duplicate output-projection key.
2. Load with `strict=False` and explicitly restore the shared reference.

```python
state_dict = {
    key: value
    for key, value in model.state_dict().items()
    if key != "final_linear_layer.weight"
}
save_file(state_dict, filepath)
```

```python
model.load_state_dict(state_dict, strict=False)
model.final_linear_layer.weight = model.token_embedding.weight
```

The result stores one tensor, wastes no checkpoint space, and restores the original model structure after loading.

## 2. Ten-epoch run: overfitting and repetitive generation

The second run trained successfully but produced degenerate output such as repeated instances of “love,” “marry,” and “followed.”

The configuration used for that experiment was:

| Parameter | Value |
| --- | ---: |
| Transformer blocks | 1 |
| Block size | 32 |
| Epochs | 10 |
| Batch size | 64 |

The dataloaders contained 30,299 training batches, 3,788 validation batches, and 3,788 test batches.

| Epoch | Train loss | Validation loss | Learning rate |
| ---: | ---: | ---: | ---: |
| 1 | 5.9298 | **4.7935** | 2.93e-04 |
| 2 | 4.0938 | 5.1153 | 2.72e-04 |
| 3 | 3.4506 | 5.3138 | 2.40e-04 |
| 4 | 3.2315 | 5.4557 | 2.00e-04 |
| 5 | 3.0395 | 5.5502 | 1.55e-04 |
| 6 | 2.8994 | 5.7220 | 1.10e-04 |
| 7 | 2.7262 | 5.7582 | 6.98e-05 |
| 8 | 2.6351 | 5.8352 | 3.77e-05 |
| 9 | 2.4971 | 5.9216 | 1.71e-05 |
| 10 | 2.3919 | 5.9922 | 1.00e-05 |

### Diagnosis

Validation loss began diverging from training loss after epoch 1. The final train/validation gap was approximately 3.60, and the epoch-10 validation result was the worst in the run. The model had memorized the training corpus instead of improving its ability to generalize.

Four factors contributed to the poor generation:

1. **The final checkpoint was used instead of the best checkpoint.** Every epoch after the first worsened validation loss, so inference used the most overfit version.
2. **One Transformer block provided very little depth.** It could learn shallow local statistics but had limited capacity for longer structure.
3. **A 32-token training window was too short.** The model never learned behavior over a longer sentence or paragraph context.
4. **Generation had no repetition penalty.** Top-k sampling and temperature alone did not prevent a high-probability token loop.

### Fixes

The trainer now saves a checkpoint only when validation loss improves. Early stopping terminates training after three consecutive epochs without improvement:

```python
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    epochs_without_improvement = 0
    save_model()
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
    break
```

The native generator also accepts `--repetition_penalty` and defaults to 1.3. For tokens already present in the context, it reduces positive logits and makes negative logits more negative before top-k filtering.

## 3. Proposed third-run configuration

The historical plan for a third training run was:

| Parameter | Before | Proposed | Reason |
| --- | ---: | ---: | --- |
| Transformer blocks | 1 | 6 | Add representational depth while remaining more practical than a 12-block GPT-2 Small-style model. |
| Block size | 32 | 128 | Expose the model to sentence- and paragraph-level context. |
| Dropout | 0.1 | 0.2 | Offset the overfitting risk from greater capacity. |
| Batch size | 64 | 32 | Reduce memory use because each sample contains four times as many tokens. |

Because `block_size` is baked into the tensor shapes, the dataset must be regenerated after changing it:

```bash
python corpus/data/pretraining_data/prepare_dataset.py
python model/gpt/train_gpt.py
```

The expected outcome was earlier convergence, a checkpoint with validation loss below 4.7935, and more coherent text with fewer repetition loops. These were expectations, not recorded results.

## 4. Dataset preparation OOM

Running `bash run.sh` once exhausted approximately 30 GB of RAM during dataset preparation. The earlier implementation created sliding windows by slicing the complete token sequence and appending each slice to Python lists. That approach materialized roughly `O(number_of_tokens × block_size)` Python objects, whose overhead dominated memory use.

### Fix

`prepare_dataset.py` was changed to convert the token stream to one PyTorch tensor and use `unfold`:

```python
tokens_tensor = torch.tensor(encoded_texts_ids, dtype=torch.long)
X_tensor = tokens_tensor[:-1].unfold(0, block_size, 1)
y_tensor = tokens_tensor[1:].unfold(0, block_size, 1)
```

`unfold` produces sliding-window views over shared storage instead of creating a Python list for every window.

The resulting tensors can be non-contiguous. The training loss calculation therefore uses `reshape(-1)` instead of `view(-1)`, since `view` requires a compatible contiguous layout.

**Lesson:** use tensor-native windowing rather than Python lists for large language-model datasets.

## 5. Expected initial cross-entropy loss

For a vocabulary of size `V`, a model whose initial prediction is close to a uniform distribution should have a cross-entropy loss near:

```text
ln(V)
```

With the configured vocabulary size of 32,000, this reference value is approximately `ln(32000) = 10.37`. A substantially different initial loss is a useful signal to inspect initialization, target shifting, masking, vocabulary alignment, and whether the reported value is truly the first batch or an epoch average.

## 6. Inference requirements should shape the model design

The original model used `nn.MultiheadAttention`, whose packed projection interface does not directly expose the per-layer K and V tensors needed by the custom cache manager. Without a KV cache, autoregressive generation recomputes the full sequence for every new token.

The learned weights did not need to be discarded. The packed `in_proj_weight` and `in_proj_bias` can be split into Q, K, and V projections, while the output-projection weights are copied directly. A cache-aware model can then reuse those trained weights and add separate prefill/decode behavior with the correct position offsets.

For future model work, inference requirements should be planned before training, especially:

- KV-cache support;
- padding and causal attention masks;
- checkpoint/export compatibility;
- stable weight names;
- generation API contracts; and
- compatibility with Transformers or serving engines such as vLLM.

## 7. KV-cache inference implementation: mistakes and fixes

### 7.1 Applying a square causal mask during cached decoding

**Bug:** `causal_mask[:seq_len, :seq_len]` was applied unconditionally. During single-token decoding, the query length is 1 while the keys include the entire cached prefix, so a `[1, 1]` mask does not match `[1, cached_length + 1]` attention scores.

**Fix:** skip the causal mask when `past_kv` exists for single-token decoding. The new token is allowed to attend to every past key and its own key; no future keys are present.

### 7.2 Resetting position IDs to zero

**Bug:** every single generated token received position 0, destroying its positional context.

**Fix:** calculate `past_length` from the cache and create position IDs beginning at that offset.

### 7.3 Loading the model inside the generation loop

**Bug:** `AutoModelForCausalLM.from_pretrained` was invoked for token generation, adding model-loading latency repeatedly.

**Fix:** load the model once in `KVCacheManager.__init__` and reuse it.

### 7.4 Hashing the growing full sequence for prefix lookup

**Bug:** the entire token sequence was used as a cache key on every iteration. Since the sequence grows by one token, the key never matched the cached prefix.

**Fix:** look up the relevant prefix (`input_tokens[:, :-1]`) or otherwise use a stable prefix key.

### 7.5 Passing the full sequence together with a cache

**Bug:** the full `input_tokens` tensor was passed with `past_key_values` during every decode step. All of those tokens were appended to the cache repeatedly, causing explosive memory growth.

**Fix:** after prefill, pass only the newest token to the model.

### 7.6 Running prefill twice

**Bug:** the cache-miss path called `prefill_kv_cache()` and then ran `model(input_tokens)` again.

**Fix:** run the prompt once and store the cache returned by that same forward pass.

### 7.7 Returning final-step logits instead of the generated tokens

**Bug:** the function returned the final `output`, which contains logits only for the last cached decode input.

**Fix:** return the accumulated `input_tokens` tensor, which contains the prompt and all generated tokens.

## 8. Why converted attention can run with garbage output

Replacing `nn.MultiheadAttention` with explicit layers such as:

```python
self.query_proj = nn.Linear(config.d_model, config.d_model)
self.key_proj = nn.Linear(config.d_model, config.d_model)
self.value_proj = nn.Linear(config.d_model, config.d_model)
```

does not cause a shape error. PyTorch initializes the new layers randomly, and all matrix dimensions remain valid, so the forward pass completes normally.

The Hugging Face loader matches state-dict keys by name. The old checkpoint contains names such as `multihead_attention.in_proj_weight`, while the new architecture expects `query_proj.weight`, `key_proj.weight`, and `value_proj.weight`. With a non-strict or tolerant loading path, those unmatched layers remain randomly initialized after a warning.

The model therefore runs but produces unrelated or nonsensical text. The trained packed values must be injected explicitly:

```python
W_q, W_k, W_v = old_mha.in_proj_weight.chunk(3, dim=0)
b_q, b_k, b_v = old_mha.in_proj_bias.chunk(3, dim=0)

new_mha.query_proj.weight.data.copy_(W_q)
new_mha.query_proj.bias.data.copy_(b_q)
new_mha.key_proj.weight.data.copy_(W_k)
new_mha.key_proj.bias.data.copy_(b_k)
new_mha.value_proj.weight.data.copy_(W_v)
new_mha.value_proj.bias.data.copy_(b_v)
new_mha.out_proj.weight.data.copy_(old_mha.out_proj.weight)
new_mha.out_proj.bias.data.copy_(old_mha.out_proj.bias)
```

Once the weights are copied correctly, the explicit-attention implementation should be checked for numerical equivalence with the original implementation before further serving optimizations are added.

## 9. Tied weights when saving the Hugging Face conversion

The native-checkpoint solution in section 1 omits the duplicate output key. The Hugging Face `save_pretrained()` conversion path uses a different practical approach because it owns the full model serialization flow.

Before saving the converted model, the tied output weight is cloned:

```python
new_model.final_linear_layer.weight = torch.nn.Parameter(
    new_model.token_embedding.weight.clone()
)
```

Safetensors then sees two equal but separately allocated tensors and can serialize them. When the custom model is loaded later, its `tie_weights()` method points `final_linear_layer.weight` back to `token_embedding.weight` so memory sharing is restored.

Both strategies solve the same aliasing problem:

- **Native checkpoint:** omit the duplicate key, load non-strictly, and re-tie explicitly.
- **Hugging Face conversion:** temporarily clone for serialization and re-tie through the model lifecycle.

## Summary of reusable lessons

1. Track validation loss and save the best checkpoint, not merely the final one.
2. Compare the initial language-model loss with `ln(vocabulary_size)` as a sanity check.
3. Use tensor-native operations such as `unfold` for large sliding-window datasets.
4. Treat weight tying as an explicit save/load contract.
5. Preserve trained weights exactly when changing module structure.
6. Verify logits before optimizing serving behavior.
7. Separate prompt prefill from single-token decode.
8. Apply position offsets and causal masks according to cache state.
9. Pass only new tokens after prefill and keep each request's cache isolated.
10. Profile and validate correctness before adding batching or custom kernels.

