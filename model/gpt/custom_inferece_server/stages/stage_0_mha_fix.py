from __future__ import annotations

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
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        q = self.query_proj(hidden_states)
        k = self.key_proj(hidden_states)
        v = self.value_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.number_of_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.number_of_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.number_of_head, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_score = attn_score + self.causal_mask[:seq_len, :seq_len]
        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = torch.matmul(attn_score, v)
        attn_score = attn_score.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_score)
        


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        ln1 = self.layer_norm_1(x)
        attn_out = self.multihead_attention(ln1)
        x = x + self.mha_drop(attn_out)
        ln2    = self.layer_norm_2(x)
        ff_out = self.ffn_2(self.gelu(self.ffn_1(ln2)))
        return x + self.ffn_drop(ff_out)

class GPTCustomForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GPTCustomConfig

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
        **kwargs,
    ) -> CausalLMOutput:
        batch_size, seq_len = input_ids.shape
        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = self.token_embedding(input_ids) + self.positional_encoding(position_ids)
        x = self.emb_dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.final_linear_layer(self.layer_norm_final(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)

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
