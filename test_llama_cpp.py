import base64
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

LLM_GGUF = "../my_ministral3b/my_gguf_Q4_K_M.gguf"
MMPROJ_GGUF = "../my_ministral3b/mmproj-F16.gguf"
IMAGE_PATH = "/captures/capture.jpg"

def image_to_data_uri(path: str) -> str:
    b = Path(path).read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    # 大多數範例用 data URI 方式把圖片塞進 messages
    return f"data:image/jpeg;base64,{b64}"

chat_handler = Llava15ChatHandler(
    clip_model_path=MMPROJ_GGUF,
)

llm = Llama(
    model_path=LLM_GGUF,
    chat_handler=chat_handler,
    n_ctx=4096,        # 圖片 embedding 會吃 context，通常要比純文字大 :contentReference[oaicite:2]{index=2}
    logits_all=True,   # Llava 常見需求：需要 logits_all 才能正常跑 vision :contentReference[oaicite:3]{index=3}
    n_gpu_layers=-1,   # Jetson 上你可視 VRAM 調整；-1 表示盡量上 GPU（若你編譯/後端支援）
)

img_uri = image_to_data_uri(IMAGE_PATH)

# 重要：messages 裡把圖片用 data URI 放進去（常見模式）
resp = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "請描述這張圖片，並用 JSON 回答：{name,color,count}"},
                {"type": "image_url", "image_url": {"url": img_uri}},
            ],
        },
    ],
    temperature=0,
)

print(resp["choices"][0]["message"]["content"])
