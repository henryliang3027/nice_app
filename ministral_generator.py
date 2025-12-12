import os
from PIL import Image
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import time
import anyio
import cv2
import torch

class MinistralGenerator:
    def __init__(self):
        model_id = "unsloth/Ministral-3-3B-Instruct-2512"
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_id,
            load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers     = True, # False if not finetuning vision layers
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers

            r = 32,           # The larger, the higher the accuracy, but might overfit
            lora_alpha = 32,  # Recommended alpha == r at least
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )

        FastVisionModel.for_inference(self.model) # Enable for inference!


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
        pil_image = self.resize_image(pil_image)

        print('image converted')

        messages = [
            # {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]

        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = self.tokenizer(
            pil_image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")


        print('self.tokenizer')

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=False,
                use_cache=True,
                temperature=1.0,
                min_p=0.1,
            )

        # 從 generated_ids 取出模型輸出的文字
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]


        # from transformers import TextStreamer
        # text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        # print('text_streamer')
        # _ = self.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000,
        #                 use_cache = True, temperature = 1.5, min_p = 0.1)
        print(f'response:{response}')

        
        return response

