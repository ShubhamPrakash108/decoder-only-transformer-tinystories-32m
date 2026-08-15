import os
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

# Determine the absolute path to the project root directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Append the project root to sys.path so we can import custom modules (like utils)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.file_utils import load_yaml

# Construct the path to the YAML configuration file and load it
config_path = os.path.join(root_dir, 'config', 'config.yml')
GPT_CONFIG = load_yaml(config_path)

class GPTBlock(nn.Module):
    """
    A single transformer block for the GPT model.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        # Layer normalization applied before self-attention (Pre-LN architecture)
        self.layer_norm_1 = nn.LayerNorm(config['d_model'])
        self.layer_norm_2 = nn.LayerNorm(config['d_model'])
        
        # Multi-head attention mechanism to capture dependencies across the sequence
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["num_heads"],
            batch_first=True
        )
        
        # Activation function for the feed-forward network
        self.gelu = nn.GELU()
        
        # Two-layer feed-forward network (MLP) mapping d_model -> 4*d_model -> d_model
        self.ffn_1 = nn.Linear(config['d_model'], config['d_model'] * 4)
        self.ffn_2 = nn.Linear(config['d_model'] * 4, config['d_model'])
        
        # Dropout layers for regularization
        self.mha_dropout = nn.Dropout(config['dropout'])
        self.ffn_dropout = nn.Dropout(config['dropout'])
        
        # Register the causal mask to ensure the model only attends to past tokens
        # Using register_buffer ensures the mask is moved to the correct device (CPU/GPU) along with the model
        self.register_buffer(
            'causal_mask',
            torch.triu(
                torch.full((config['max_seq_len'], config['max_seq_len']), float('-inf')),
                diagonal=1
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPTBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        _, seq_len, _ = x.size()
        
        # Self-Attention Sub-layer
        layer_norm_1_out = self.layer_norm_1(x)
        
        # Compute self-attention with the causal mask to prevent peeking into the future
        mha_out, _ = self.multihead_attention(
            query=layer_norm_1_out,
            key=layer_norm_1_out,
            value=layer_norm_1_out,
            attn_mask=self.causal_mask[:seq_len, :seq_len]
        )
        # First residual connection
        x = x + self.mha_dropout(mha_out)

        # Feed-Forward Network Sub-layer
        layer_norm_2_out = self.layer_norm_2(x)
        ffn_1_out = self.ffn_1(layer_norm_2_out)
        gelu_out = self.gelu(ffn_1_out)
        ffn_2_out = self.ffn_2(gelu_out)
        
        # Second residual connection
        out = x + self.ffn_dropout(ffn_2_out)

        return out


class GPTModel(nn.Module):
    """
    The main GPT model architecture.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        # Embeddings for token position in the sequence and the actual token vocabulary
        self.positional_encoding = nn.Embedding(config['max_seq_len'], config['d_model'])
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.emb_dropout = nn.Dropout(config['dropout'])
        
        # A list containing the sequential Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config['number_of_transformer_block'])
        ])
        
        # Final layer normalization before predicting the next token
        self.layer_norm_final = nn.LayerNorm(config['d_model'])
        
        # Linear layer mapping the hidden dimension back to vocabulary space (logits)
        self.final_linear_layer = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

        # Weight tying: The embedding matrix and final output matrix share weights to save parameters
        self.final_linear_layer.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPTModel.
        
        Args:
            input_ids (torch.Tensor): Tensor of token IDs, shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = input_ids.shape
        
        # Generate position IDs from 0 to seq_len-1 for the batch
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Compute position and token embeddings
        position_embeddings = self.positional_encoding(position_ids)
        token_embeddings = self.token_embedding(input_ids)
        
        # Combine embeddings and apply dropout
        x = position_embeddings + token_embeddings
        x = self.emb_dropout(x)

        # Sequentially pass the input through all transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply final layer normalization and compute logits
        x = self.layer_norm_final(x)
        logits = self.final_linear_layer(x)

        return logits


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# if __name__ == "__main__":
#     gpt2_model = GPTModel(GPT_CONFIG)
#     total_params = count_parameters(gpt2_model)
#     print(f"TOTAL PARAMETERS: {total_params}")
#     # Expected output: TOTAL PARAMETERS: 124439808