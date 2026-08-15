from model.gpt.custom_inferece_server.stages.stage_2_kv_cache import KVCacheManager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
from fastapi import HTTPException


app = FastAPI(title = "Custom GPT Server")

KV_cache = KVCacheManager()


class CompletionRequest(BaseModel):
    model: str = "gpt-custom"
    prompt: str
    max_tokens: Optional[int] = 32

@app.post("/v1/completions")
def completions(req: CompletionRequest):
    try:
        input_ids = KV_cache.tokenizer.encode(req.prompt, return_tensors='pt')
        device = next(KV_cache.model.parameters()).device
        input_ids = input_ids.to(device)
        response_tensor = KV_cache.generate_next_token(input_ids, req.max_tokens)
        generated_text = KV_cache.tokenizer.decode(response_tensor[0][input_ids.shape[1]:], skip_special_tokens=True)

        return {
            "id": "cmpl-custom",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }]
        }
    except Exception as e:
        import traceback
        traceback.print_exc() # THIS is what prints the full red error to your terminal!
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(" Starting Custom GPT Inference Server...")
    print(" OpenAI API endpoint available at: http://localhost:8000/v1/completions\n")
    uvicorn.run(app, host="[IP_ADDRESS]", port=8000)