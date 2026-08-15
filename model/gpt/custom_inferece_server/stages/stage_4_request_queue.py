from model.gpt.custom_inferece_server.stages.stage_2_kv_cache import KVCacheManager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
from fastapi import HTTPException
import asyncio


app = FastAPI(title = "Custom GPT Server")

KV_cache = KVCacheManager()

queue = asyncio.Queue()

class CompletionRequest(BaseModel):
    model: str = "gpt-custom"
    prompt: str
    max_tokens: Optional[int] = 32

async def llm_worker():
    while True:
        prompt, max_tokens, future = await queue.get()
        try:
            input_ids = KV_cache.tokenizer.encode(prompt, return_tensors='pt')
            device = next(KV_cache.model.parameters()).device
            input_ids = input_ids.to(device)
            response_tensor = await asyncio.to_thread(KV_cache.generate_next_token, input_ids, max_tokens)
            generated_text = KV_cache.tokenizer.decode(response_tensor[0][input_ids.shape[1]:], skip_special_tokens=True)
            future.set_result(generated_text)
        except Exception as e:
            future.set_exception(e)
        finally:
            queue.task_done()  # always signals the queue that this slot is free, even if generation crashed

@app.on_event("startup")  # FastAPI runs this function automatically when the server boots
async def startup_event():
    asyncio.create_task(llm_worker())  # starts llm_worker as a background task so it doesn't block the web server


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    try:
        future = asyncio.get_event_loop().create_future()  # unique empty "return box" — the worker will fill this with the generated text
        await queue.put((req.prompt, req.max_tokens, future))
        generated_text = await future
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