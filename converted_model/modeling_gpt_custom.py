from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

class CustomCausalMHA(nn.Module):
    def __init__(self, config: GPTCustomConfig):
        super().__init__()
        self.number_of_head = config.num_heads
        self.d_model        = config.d_model
        self.head_dim       = config.d_model // config.num_heads
        self.scale          = 1 / (self.head_dim ** 0.5)
        self.query_proj     = nn.Linear(config.d_model, config.d_model)
        self.key_proj       = nn.Linear(config.d_model, config.d_model)
        self.value_proj     = nn.Linear(config.d_model, config.d_model)
        self.out_proj       = nn.Linear(config.d_model, config.d_model)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((config.max_seq_len, config.max_seq_len), float("-inf")),
                diagonal=1,
            ),
        )
    
    def forward(self, hidden_states: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # [AI-WRITTEN]
        batch_size, seq_len, _ = hidden_states.size()
        q = self.query_proj(hidden_states)
        k = self.key_proj(hidden_states)
        v = self.value_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.number_of_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.number_of_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.number_of_head, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if past_kv is not None:
            k_cache, v_cache = past_kv
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # NOTE: MATH OF CAUSAL MASKING WITH KV CACHE:
        # q shape: [batch, heads, seq_len, head_dim]
        # k shape: [batch, heads, past_len + seq_len, head_dim]
        # attn_score shape = q @ k.T = [batch, heads, seq_len, past_len + seq_len]
        # 
        # Case 1: Prefill (past_kv is None)
        #   seq_len = N, past_len = 0. attn_score is [N, N]. 
        #   Token 'i' cannot look at future token 'j' (where j > i). We MUST apply the [N, N] causal mask.
        #
        # Case 2: Generation (past_kv is NOT None)
        #   seq_len = 1, past_len = L. attn_score is [1, L + 1].
        #   We have 1 query token (the current token). It is allowed to look at ALL past L tokens 
        #   and itself. There are no "future" tokens in the K matrix to hide! 
        #   Therefore, no causal mask is needed. (Plus, adding a [1, 1] mask to a [1, L+1] matrix would crash).
        if past_kv is None:
            attn_score = attn_score + self.causal_mask[:seq_len, :seq_len]

        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = torch.matmul(attn_score, v)
        attn_score = attn_score.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_score), k, v
        


class GPTCustomConfig(PretrainedConfig):
    model_type = "gpt-custom"
    attribute_map = {
        "num_hidden_layers":  "number_of_transformer_block",
        "hidden_size":        "d_model",
        "num_attention_heads": "num_heads",
    }
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        num_heads: int = 8,
        number_of_transformer_block: int = 6,
        max_seq_len: int = 1024,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size                  = vocab_size
        self.d_model                     = d_model
        self.num_heads                   = num_heads
        self.number_of_transformer_block = number_of_transformer_block
        self.max_seq_len                 = max_seq_len
        self.dropout                     = dropout

class _GPTBlock(nn.Module):
    def __init__(self, config: GPTCustomConfig) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)
        self.multihead_attention = CustomCausalMHA(config)
        self.gelu      = nn.GELU()
        self.ffn_1     = nn.Linear(config.d_model, config.d_model * 4)
        self.ffn_2     = nn.Linear(config.d_model * 4, config.d_model)
        self.mha_drop  = nn.Dropout(config.dropout)
        self.ffn_drop  = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((config.max_seq_len, config.max_seq_len), float("-inf")),
                diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, seq_len, _ = x.size()
        ln1 = self.layer_norm_1(x)
        attn_out, k, v = self.multihead_attention(ln1, past_kv=past_kv)
        x = x + self.mha_drop(attn_out)
        ln2    = self.layer_norm_2(x)
        ff_out = self.ffn_2(self.gelu(self.ffn_1(ln2)))
        return x + self.ffn_drop(ff_out), k, v

class GPTCustomForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GPTCustomConfig
    _tied_weights_keys = ["final_linear_layer.weight"]

    def __init__(self, config: GPTCustomConfig) -> None:
        super().__init__(config)
        self.token_embedding      = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding  = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout          = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList(
            [_GPTBlock(config) for _ in range(config.number_of_transformer_block)]
        )
        self.layer_norm_final   = nn.LayerNorm(config.d_model)
        self.final_linear_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.final_linear_layer.weight = self.token_embedding.weight
        self.config.is_decoder = True
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: dict | None = None,
        **kwargs
    ) -> tuple[CausalLMOutput, dict]:
        batch_size, seq_len = input_ids.shape
        
        
        # Calculate how many tokens are already in the cache. 
        # When generating token-by-token, we must know the past sequence length 
        # so we can give the new token the correct positional embedding offset.
        past_length = 0 
        if past_key_values is not None:
            # We grab the cache for the first block (any block works)
            first_block_cache = list(past_key_values.values())[0]
            # first_block_cache[0] is 'k', its shape is (batch, num_heads, seq_len, head_dim)
            # So dim=2 is the sequence length of the past cache
            past_length = first_block_cache[0].size(2)
            
        # Shift position IDs by the past_length so new tokens get correct positional embeddings.
        # Example: if cache has 10 tokens, the 1 new token we pass in must have position_id = 10, not 0.
        position_ids = (
            torch.arange(past_length, past_length + seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = self.token_embedding(input_ids) + self.positional_encoding(position_ids)
        x = self.emb_dropout(x)
        
        cache = {}
        for block in self.transformer_blocks:
            # Grab the specific cache for this particular block
            block_past_kv = past_key_values.get(block, None) if past_key_values is not None else None 
            
            # Pass it down into the block
            x, k, v = block(x, past_kv=block_past_kv)
            
            if use_cache:
                cache[block] = (k, v)
                
        logits = self.final_linear_layer(self.layer_norm_final(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits), cache

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embedding = value

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        return {"input_ids": input_ids}

    def tie_weights(self, **kwargs) -> None:
        self.final_linear_layer.weight = self.token_embedding.weight
