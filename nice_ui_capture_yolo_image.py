from nicegui import ui
from opencc import OpenCC
import torch
import cv2
import base64
import os
import httpx
import random
import numpy as np
from ollama import chat
import anyio
from utils import cv_to_base64
from datetime import datetime
from openai import OpenAI

# measure elapsed time
import time


SYSTEM_PROMPT = """你是零售貨架的商品統計助手。

輸出格式（CSV）：
商品名稱,顏色,數量

規則：

1. 每個商品一行
2. 使用最短的中文商品名稱
3. 顏色用中文單字
4. 數量只用整數

範例：
冷山茶王,藍,2
飲冰室茶集,綠,1
可口可樂,紅,3"""


class WebcamDisplay:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.timer = None
        self.current_base64_frame = None
        self.current_mat_frame = None
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
        """Initialize = self.find_cameras_id()
            print(f"c and start the webcam"""
        if self.cap is None:
            camera_id = self.find_cameras_id()
            print(f"camera id: {camera_id}")
            self.cap = cv2.VideoCapture(camera_id)  # 0 is the default webcam
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
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

                self.current_mat_frame = frame

                
                
                _, imencode_image = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(imencode_image)
                base64_image_string = 'data:image/jpg;base64,' + base64_image.decode('ascii')

                self.current_base64_frame = base64_image_string


                return base64_image_string
        return None

    def capture_base64_image(self):
        """Save the current frame as a JPEG image"""
        if self.current_base64_frame is not None:
            return self.current_base64_frame
        return None

    def capture_mat_image(self):
        """Save the current frame as a JPEG image"""
        if self.current_mat_frame is not None:
            return self.current_mat_frame
        return None


# Create webcam instance
webcam = WebcamDisplay()

# Convert Simplified Chinese to Traditional Chinese
cc = OpenCC('s2t')

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="no-key-needed"  # 本地通常不驗證，填任意字串即可
)

# Create the UI
@ui.page('/')
def main_page():
    # Create an interactive image element
    img = ui.interactive_image().classes('w-full max-w-3xl border-4 border-gray-300 rounded-lg')

    # Status label for capture feedback
    status_label = ui.label('').classes('text-sm')


    # API response display
    api_response_label = ui.label('').classes('text-sm bg-gray-100 rounded whitespace-pre-line')

    answer_label  = ui.label('').classes('text-3xl text-green-700 rounded whitespace-pre-line')
    time_elapse_label = ui.label('').classes('text-sm text-green-700 rounded whitespace-pre-line')

    answer_label.visible = False
    time_elapse_label.visible = False

    ori_answer_label  = ui.label('').classes('text-3xl text-green-700 rounded whitespace-pre-line')
    ori_time_elapse_label = ui.label('').classes('text-sm text-green-700 rounded whitespace-pre-line')

    ori_answer_label.visible = False
    ori_time_elapse_label.visible = False


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
        time_elapse_label.set_text('')
        
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
        time_elapse_label.set_text('')
        answer_label.visible = False
        time_elapse_label.visible = False

    async def inference(base64_image, question):
        return await anyio.to_thread.run_sync(
            inference_sync, 
            base64_image, 
            question
        )

    def inference_sync(base64_image, question):
        resppnse = client.chat.completions.create(
            model="ministral_custom",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ]},
            ],
            temperature=0,
        )

        return resppnse.choices[0].message.content
    

    async def inference_from_image_path(image_path, question):
        return await anyio.to_thread.run_sync(
            inference_sync_from_image_path, 
            image_path, 
            question
        )

    def inference_sync_from_image_path(image_path, question):
        response = chat(
            model='my_ministral3b:latest',
            messages=[
                {
                'role': 'user',
                'content': question,
                'images': [image_path],
                }
            ],
        )

        return response.message.content


    async def capture_frame():
        """Capture and save the current frame"""
        capture_btn.disable()
        ori_answer_label.visible = False
        ori_time_elapse_label.visible = False
        answer_label.visible = False
        time_elapse_label.visible = False
        api_response_label.visible = True
        current_base64_frame = webcam.capture_base64_image()
        current_mat_frame = webcam.capture_mat_image()

        if len(current_base64_frame and current_mat_frame) > 0:
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
        print(question)

        
        # Show loading status
        api_response_label.set_text('⏳ Processing...')
        api_response_label.classes('text-blue-600')

        # Call API
        # response = await webcam.call_api(filename, question)
        
        # start = time.perf_counter()
        # inference
        # response = await qwenGenerator.inference(current_frame, question)




        # response = await inference(current_base64_frame, question)

        # elapsed = time.perf_counter() - start
        # print(f"Elapsed: {elapsed:.4f} seconds")

        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d-%H:%M:%S")
        image_save_path = f'captures/capture_{formatted}.jpg'
        cv2.imwrite(image_save_path, current_mat_frame)

        # ori_start = time.perf_counter()
        # original_model_response = await inference_from_image_path(image_save_path, question)
        # ori_elapsed = time.perf_counter() - ori_start

        if True:
            api_response_label.visible = False
            answer_label.visible = True
            time_elapse_label.visible = True
            ori_answer_label.visible = True
            ori_time_elapse_label.visible = True
            start = time.perf_counter()
            # response = cc.convert(response)
            convert_elapsed = time.perf_counter() - start
            print(f"convert elapsed: {convert_elapsed:.4f} seconds")
            api_response_label.set_text('')

            # time_elapse_label.set_text(f"{elapsed:.4f} 秒")
            # answer_label.set_text(response)

            # ori_time_elapse_label.set_text(f"{ori_elapsed:.4f} 秒")
            # ori_answer_label.set_text(original_model_response)

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
    with ui.column().classes('w-full max-w-2xl gap-2'):
        question_input = ui.textarea(
            label='Question', 
            placeholder='統計圖中的商品',
            value='統計圖中的商品'
        ).classes('w-full')

        keyword_btn1 = ui.button('統計圖中的商品', on_click=lambda: fill_keyword("統計圖中的商品")).classes('bg-blue-500')
        keyword_btn1 = ui.button('圖中有幾瓶冷山茶王', on_click=lambda: fill_keyword("圖中有幾瓶冷山茶王")).classes('bg-blue-500')
        keyword_btn2 = ui.button('圖中有幾瓶綠色包裝的麥香飲料', on_click=lambda: fill_keyword("圖中有幾瓶綠色包裝的麥香飲料")).classes('bg-blue-500')
        keyword_btn3 = ui.button('圖中有幾瓶包裝為麥香綠茶', on_click=lambda: fill_keyword("圖中有幾瓶包裝為麥香綠茶")).classes('bg-blue-500')
        keyword_btn4 = ui.button('圖中有幾瓶有藍色瓶蓋的飲料', on_click=lambda: fill_keyword("圖中有幾瓶有藍色瓶蓋的飲料")).classes('bg-blue-500')
        keyword_btn5 = ui.button('圖中有幾瓶包裝上標有 Green Tea', on_click=lambda: fill_keyword("圖中有幾瓶包裝上標有 Green Tea")).classes('bg-blue-500')
        
    ui.label('Click "Start Webcam" to begin streaming').classes('text-sm text-gray-600 mt-2')







# Run the app
ui.run(title='Webcam Display', port=8881, reload=False)



