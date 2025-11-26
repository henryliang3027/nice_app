# 合併並儲存（只需要做一次）
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

base_model_id = "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
adapter_path = "./checkpoints/checkpoint-50240"
merged_path = "./checkpoints/merged_model"  # 儲存位置

# 載入
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

peft_model = PeftModel.from_pretrained(base_model, adapter_path)

# 合併
merged_model = peft_model.merge_and_unload()

# 儲存
merged_model.save_pretrained(merged_path)

# Processor 也一起存
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(merged_path)

print(f"Merged model saved to {merged_path}")