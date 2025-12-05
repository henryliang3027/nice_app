from nicegui import ui
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datetime import datetime
from io import BytesIO
from PIL import Image
from opencc import OpenCC
import torch
import cv2
import base64
import os
import httpx
import re
import anyio
import random
import numpy as np

# measure elapsed time
import time

class QwenGenerator:
    def __init__(self):
        self.base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.adapter_path = "./checkpoints/checkpoint-50240"
        
        self.system_prompt = (
            "你是一個專業的商品計數助手。\n"
            "任務流程必須嚴格遵守：\n"
            "1. 嚴格比對使用者指定的商品名稱與圖片中的商品，名稱不完全一致視為不存在。\n"
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
        self.processor = AutoProcessor.from_pretrained(self.base_model_id, use_fast=True, padding_side="left")

        print("Loading base model and LoRA adapter...")
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        
        # self.finetuned_model = torch.compile(self.finetuned_model, mode="reduce-overhead")
        

        print("Loading LoRA adapter...")
        self.finetuned_model = PeftModel.from_pretrained(self.base_model, self.adapter_path)
        self.finetuned_model.eval()
        print("Fine-tuned model loaded!")

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




class WebcamDisplay:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.timer = None
        self.current_frame = None
        self.api_url = 'https://gillian-unhesitative-jestine.ngrok-free.dev/inference'
        

    def find_cameras_id(self, max_tested=5):
        first_available_id = -2
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera index {i} is VALID")
                    first_available_id = i
                    break
        cap.release()
        return first_available_id


    def start_webcam(self):
        """Initialize and start the webcam"""
        if self.cap is None:
            camera_id = self.find_cameras_id()
            print(f"camera id: {camera_id}")
            self.cap = cv2.VideoCapture(camera_id)  # 0 is the default webcam
        self.is_running = True
        
    def stop_webcam(self):
        """Stop the webcam"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def get_frame(self):
        """Capture and return a frame from the webcam"""
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # # Convert BGR to RGB
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # # Convert to PIL Image
                # pil_img = Image.fromarray(frame_rgb)
                # # Convert to base64
                # buffer = BytesIO()
                # pil_img.save(buffer, format='JPEG', quality=85)
                # img_str = base64.b64encode(buffer.getvalue()).decode()

                self.current_frame = frame

                _, imencode_image = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(imencode_image)
                base64_image_string = 'data:image/jpg;base64,' + base64_image.decode('ascii')

                return base64_image_string
        return None

    def capture_image(self):
        """Save the current frame as a JPEG image"""
        if self.current_frame is not None:
            # # Create captures directory if it doesn't exist
            # os.makedirs('captures', exist_ok=True)
            
            # # Generate filename with timestamp
            # # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # filename = f'captures/capture.jpg'
            
            # # Save the image
            # cv2.imwrite(filename, self.current_frame)
            return self.current_frame
        return None


    async def call_api(self, image_path, question, job_id='6391fccd-8d6c-472f-bce0-d0747b9140fe', checkpoint=''):
        """
        Call the FastAPI inference endpoint
        
        Args:
            image_path: Path to the image file
            question: Question text to send
            job_id: Optional job ID (will generate UUID if not provided)
            checkpoint: Checkpoint parameter (default empty string)
            
        Returns:
            API response as dict or None if failed
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                with open(image_path, 'rb') as f:
                    files = {
                        'image': (os.path.basename(image_path), f, 'image/jpeg')
                    }
                    data = {
                        'question': question,
                        'job_id': job_id,
                        'checkpoint': checkpoint
                    }
                    
                    response = await client.post(
                        self.api_url,
                        files=files,
                        data=data,
                        headers={'accept': 'application/json'}
                    )
                    
                    response.raise_for_status()
                    return response.json()
                    
        except Exception as e:
            print(f"API call failed: {e}")
            return None


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Create webcam instance
webcam = WebcamDisplay()

# Create QwenGenerator instance
qwenGenerator = QwenGenerator()

# Convert Simplified Chinese to Traditional Chinese
cc = OpenCC('s2t')




# Create the UI
@ui.page('/')
def main_page():

    

    # Create an interactive image element
    img = ui.interactive_image().classes('w-full max-w-3xl border-4 border-gray-300 rounded-lg')

    # Status label for capture feedback
    status_label = ui.label('').classes('text-sm mt-2')


    # API response display
    api_response_label = ui.label('').classes('text-sm mt-2 p-2 bg-gray-100 rounded whitespace-pre-line')

    answer_label  = ui.label('').classes('text-3xl p-2 text-green-700 rounded whitespace-pre-line')
    reasoning_label = ui.label('').classes('text-sm p-2 text-green-700 rounded whitespace-pre-line')

    answer_label.visible = False
    reasoning_label.visible = False


    def update_frame(): 
        """Update the image with the latest webcam frame"""
        if webcam.is_running:
            frame = webcam.get_frame()
            if frame:
                img.source = frame
    
    def start_stream():
        """Start the webcam stream"""
        webcam.start_webcam()
        # Create a timer that calls update_frame every 33ms (~30 FPS)
        webcam.timer = ui.timer(0.033, update_frame)
        start_btn.disable()
        stop_btn.enable()
        status_label.set_text('')
        api_response_label.set_text('')
        answer_label.set_text('')
        reasoning_label.set_text('')
        
    def stop_stream():
        """Stop the webcam stream"""
        webcam.stop_webcam()
        if webcam.timer:
            webcam.timer.cancel()
            webcam.timer = None
        img.source = ''  # Clear the image
        start_btn.enable()
        stop_btn.disable()
        status_label.set_text('')
        api_response_label.set_text('')
        answer_label.set_text('')
        reasoning_label.set_text('')
        answer_label.visible = False
        reasoning_label.visible = False

    async def capture_frame():
        """Capture and save the current frame"""
        capture_btn.disable()
        answer_label.visible = False
        reasoning_label.visible = False
        api_response_label.visible = True
        current_frame = webcam.capture_image()
        if len(current_frame) > 0:
            status_label.set_text(f'✓ Image saved')
            status_label.classes('text-green-600 font-semibold')
            ui.notify(f'Image captured', type='positive')
        else:
            status_label.set_text('✗ No frame available to capture')
            status_label.classes('text-red-600')
            ui.notify('Failed to capture image', type='negative')

        # clear labels
        status_label.set_text('')
        api_response_label.set_text('')

        # Get question from input
        question = question_input.value.strip()
        if not question:
            question = '圖中有幾瓶冷山茶王'  # Default question
        
        # Show loading status
        api_response_label.set_text('⏳ Processing...')
        api_response_label.classes('text-blue-600')

        # Call API
        # response = await webcam.call_api(filename, question)
        
        start = time.perf_counter()
        # inference
        response = await qwenGenerator.inference(current_frame, question)

        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.4f} seconds")

        if response:
            start = time.perf_counter()
            reasoning = cc.convert(response['reasoning'])
            elapsed = time.perf_counter() - start
            print(f"convert elapsed: {elapsed:.4f} seconds")
            answer = response['answer']
            api_response_label.set_text('')
            api_response_label.visible = False
            answer_label.visible = True
            reasoning_label.visible = True
            answer_label.set_text(f'Answer: {answer}')
            reasoning_label.set_text(f'Reasoning: {reasoning}')
            api_response_label.classes('text-green-700 font-mono')
            ui.notify('API call successful', type='positive')
        else:
            api_response_label.set_text('✗ API call failed')
            api_response_label.classes('text-red-600')
            ui.notify('API call failed', type='negative')
        
        capture_btn.enable()


    def fill_keyword(keyword):
        question_input.value = keyword

    
    # Control buttons
    with ui.row().classes('gap-2 mt-4'):
        start_btn = ui.button('Start Webcam', on_click=start_stream).classes('bg-green-500')
        stop_btn = ui.button('Stop Webcam', on_click=stop_stream).classes('bg-red-500')
        capture_btn = ui.button('📷 Capture', on_click=capture_frame).classes('bg-blue-500')
        stop_btn.disable()

    # Question input and API call button
    with ui.column().classes('mt-4 w-full max-w-2xl gap-2'):
        ui.label('Question for API:').classes('font-semibold')
        question_input = ui.input(
            label='Question', 
            placeholder='圖中有幾瓶冷山茶王',
            value='圖中有幾瓶冷山茶王'
        ).classes('w-full')


        keyword_btn1 = ui.button('圖中有幾瓶冷山茶王', on_click=lambda: fill_keyword("圖中有幾瓶冷山茶王")).classes('bg-blue-500')
        keyword_btn2 = ui.button('圖中有幾瓶綠色包裝的麥香飲料', on_click=lambda: fill_keyword("圖中有幾瓶綠色包裝的麥香飲料")).classes('bg-blue-500')
        keyword_btn3 = ui.button('圖中有幾瓶包裝為麥香綠茶', on_click=lambda: fill_keyword("圖中有幾瓶包裝為麥香綠茶")).classes('bg-blue-500')
        keyword_btn4 = ui.button('圖中有幾瓶有藍色瓶蓋的飲料', on_click=lambda: fill_keyword("圖中有幾瓶有藍色瓶蓋的飲料")).classes('bg-blue-500')
        
        
    ui.label('Click "Start Webcam" to begin streaming').classes('text-sm text-gray-600 mt-2')







# Run the app
ui.run(title='Webcam Display', port=8080, reload=False)



