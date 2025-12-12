from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import anyio
import cv2
import torch
import re

class QwenGenerator:
    def __init__(self):
        self.base_model_id = "/home/b40351/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
        self.merged_model_path = "./checkpoints/merged_model"

        self.system_prompt = (
            "你是一個專業的商品計數助手。\n"
            "任務流程必須嚴格遵守：\n"
            "1. 嚴格比對使用者指定的商品名稱與圖片中的商品，只要商品名稱包含使用者指定的關鍵字即視為符合(英文忽略大小寫)。\n"
            "2. 若不存在指定商品，無論圖片中有多少其他商品，一律輸出：\n"
            "<think>找不到商品</think><answer>0</answer>\n"
            "3. 僅在確認存在指定商品時，才進行數量統計。\n"

            "嚴禁以其他品牌或相似商品的數量作為回答。\n"
            "嚴禁出現推理內容與答案數值不一致的情況，若判斷不存在則答案只能為 0，若判斷存在則答案不得為 0。\n"

            "推理過程放在 <think></think>，\n"
            "最終答案只允許阿拉伯數字並放在 <answer></answer> 裡。\n"
            "當問題中同時包含英文品牌名稱與中文語句時，請以繁體中文回答；品牌名稱保持原文不翻譯。\n"
           
        )


        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.merged_model_path, use_fast=True, padding_side="left", local_files_only=True)

        print("Loading base model and LoRA adapter...")
        self.finetuned_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.merged_model_path,
            dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
        )
        
        # self.finetuned_model = torch.compile(self.finetuned_model, mode="reduce-overhead")
        

        # print("Loading LoRA adapter...")
        # self.finetuned_model = PeftModel.from_pretrained(self.base_model, self.adapter_path)
        # self.finetuned_model.eval()
        # print("Fine-tuned model loaded!")

        # print("Flash Attention")
        # # Vision Encoder
        # for name, module in self.base_model.named_modules():
        #     module_type = type(module).__name__
        #     if 'Attention' in module_type and 'visual' in name:
        #         print(f"[Vision] {module_type}")
        #         break

        # # Language Model  
        # for name, module in self.base_model.named_modules():
        #     module_type = type(module).__name__
        #     if 'Attention' in module_type and 'language_model' in name:
        #         print(f"[LLM] {module_type}")
        #         break



    def parse_response(self, text):
        # 提取 <think></think> 內容
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""

        if "盒" in reasoning:
            reasoning = reasoning.replace("盒", "瓶")
        
        # 提取 <answer></answer> 內容
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""
        
        return {
            "reasoning": reasoning,
            "answer": answer
        }

    def resize_image(self, img_pil, max_size=768):
        """調整圖片大小"""
        width, height = img_pil.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return img_pil.resize((new_width, new_height), Image.BICUBIC)
        return img_pil

    async def inference(self, image, question):
        return await anyio.to_thread.run_sync(
            self.inference_sync, 
            image, 
            question
        )

    def inference_sync(self, image, question):

        # read image from image path
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        # img = Image.open(image_path).convert("RGB")
        image = self.resize_image(pil_image)

        # Create conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        # Generate with FINE-TUNED MODEL only
        print("\nGenerating response with FINE-TUNED MODEL...")
        inputs_finetuned = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.finetuned_model.device)

        with torch.no_grad():
            finetuned_output_ids = self.finetuned_model.generate(
                **inputs_finetuned,
                max_new_tokens=4096,
                do_sample=False,        # 關閉隨機抽樣
                num_beams=1,            # 關閉 beam search 隨機性
                temperature=1.0,        # 無用，但明確設定
                top_p=1.0,              # 無抽樣範圍
                repetition_penalty=1.0, # 禁止重複懲罰引入隨機
                early_stopping=True,    # 保持生成長度一致
                pad_token_id=self.processor.tokenizer.pad_token_id, # 確保 padding 一致
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode fine-tuned model response
        finetuned_generated_ids = finetuned_output_ids[:, inputs_finetuned.input_ids.shape[1]:]
        finetuned_response = self.processor.batch_decode(finetuned_generated_ids, skip_special_tokens=True)[0]

        print("\n" + "="*80)
        print("FINE-TUNED MODEL RESPONSE:")
        print("="*80)
        print(finetuned_response)

        parsed_response = self.parse_response(finetuned_response)
        return parsed_response

