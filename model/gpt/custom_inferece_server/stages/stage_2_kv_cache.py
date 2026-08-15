# Imports
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def hash_tokens(input_tokens: torch.Tensor) -> str:
    """Converts a PyTorch tensor to a stable SHA256 hash string for use as a cache key."""
    return hashlib.sha256(input_tokens.cpu().numpy().tobytes()).hexdigest()

class KVCacheManager:
    def __init__(self, ):
        self.tokenizer = AutoTokenizer.from_pretrained(r".\converted_model")
        self.model = AutoModelForCausalLM.from_pretrained(r".\converted_model", trust_remote_code=True)
        self.kv_cache = {} # Dictionary to store KV caches for different inputs
        
        # Store cache on GPU if available, otherwise fall back to CPU RAM
        self.cache_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def move_cache_to_storage(self, cache: dict) -> dict:
        """Moves a cache dict's tensors to the cache storage device (GPU or CPU RAM)."""
        return {block: (k.to(self.cache_device), v.to(self.cache_device)) for block, (k, v) in cache.items()}

    def move_cache_to_model(self, cache: dict) -> dict:
        """Moves a cache dict's tensors back to the model's device before a forward pass."""
        model_device = next(self.model.parameters()).device
        return {block: (k.to(model_device), v.to(model_device)) for block, (k, v) in cache.items()}

    def prefill_kv_cache(self, input_tokens):
        """
        Prefills the KV cache for the input tokens
        """
        _ , cache = self.model(input_tokens, use_cache=True)
        self.kv_cache[hash_tokens(input_tokens)] = self.move_cache_to_storage(cache)

    def return_kv_cache(self, input_tokens):
        """
        Prepares the KV cache for the decoded token of a request
        """
        token_hash = hash_tokens(input_tokens)
        if token_hash in self.kv_cache:
            return self.move_cache_to_model(self.kv_cache[token_hash])
        else:
            return None

    def generate_next_token(self, input_tokens, max_output_len=32):
        """
        Generates the next token for the input tokens
        """
        output = None

        if hash_tokens(input_tokens) in self.kv_cache:
            cache = self.return_kv_cache(input_tokens)
            current_token = input_tokens[:, -1:]
            for i in range(max_output_len):
                output, cache = self.model(current_token, use_cache=True, past_key_values=cache)
                next_token = output.logits.detach().argmax(dim=-1)[0][-1]
                current_token = next_token.unsqueeze(0).unsqueeze(0)
                input_tokens = torch.cat([input_tokens, current_token], dim=-1)
        else:
            output, cache = self.model(input_tokens, use_cache=True)
            self.kv_cache[hash_tokens(input_tokens)] = self.move_cache_to_storage(cache)
            
            for i in range(max_output_len):
                next_token = output.logits.detach().argmax(dim=-1)[0][-1]
                current_token = next_token.unsqueeze(0).unsqueeze(0)
                input_tokens = torch.cat([input_tokens, current_token], dim=-1)
                output, cache = self.model(current_token, use_cache=True, past_key_values=cache)
                
        return input_tokens