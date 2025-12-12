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

# measure elapsed time
import time

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
            return self.current_frame
        return None



# Create webcam instance
webcam = WebcamDisplay()

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

    async def inference(image, question):
        return await anyio.to_thread.run_sync(
            inference_sync, 
            image, 
            question
        )

    def inference_sync(image, question):
        base64_image = cv_to_base64(image)
        response = chat(
            model='ministral-3:3b',
            messages=[
                {
                'role': 'user',
                'content': question,
                'images': [base64_image],
                }
            ],
        )

        return response.message.content
    

    async def inference_from_image_path(image_path, question):
        return await anyio.to_thread.run_sync(
            inference_sync_from_image_path, 
            image_path, 
            question
        )

    def inference_sync_from_image_path(image_path, question):
        response = chat(
            model='ministral-3:3b',
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
        print(question)

        
        # Show loading status
        api_response_label.set_text('⏳ Processing...')
        api_response_label.classes('text-blue-600')

        # Call API
        # response = await webcam.call_api(filename, question)
        
        start = time.perf_counter()
        # inference
        # response = await qwenGenerator.inference(current_frame, question)
        # image_save_path = 'captures/capture.jpg'
        # cv2.imwrite(image_save_path, current_frame)

        response = await inference(current_frame, question)

        elapsed = time.perf_counter() - start
        print(f"Elapsed: {elapsed:.4f} seconds")

        if response:
            start = time.perf_counter()
            response = cc.convert(response)
            elapsed = time.perf_counter() - start
            print(f"convert elapsed: {elapsed:.4f} seconds")
            api_response_label.set_text('')
            api_response_label.visible = False
            answer_label.visible = True
            reasoning_label.visible = True
            answer_label.set_text(response)
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
        question_input = ui.textarea(
            label='Question', 
            placeholder='統計圖中的商品，回答格式：商品名稱，顏色特徵：數量',
            value='統計圖中的商品，回答格式：商品名稱，顏色特徵：數量'
        ).classes('w-full')

        keyword_btn1 = ui.button('統計圖中的商品，回答格式：商品名稱，顏色特徵：數量', on_click=lambda: fill_keyword("統計圖中的商品，回答格式：商品名稱，顏色特徵：數量")).classes('bg-blue-500')
        keyword_btn1 = ui.button('圖中有幾瓶冷山茶王', on_click=lambda: fill_keyword("圖中有幾瓶冷山茶王")).classes('bg-blue-500')
        keyword_btn2 = ui.button('圖中有幾瓶綠色包裝的麥香飲料', on_click=lambda: fill_keyword("圖中有幾瓶綠色包裝的麥香飲料")).classes('bg-blue-500')
        keyword_btn3 = ui.button('圖中有幾瓶包裝為麥香綠茶', on_click=lambda: fill_keyword("圖中有幾瓶包裝為麥香綠茶")).classes('bg-blue-500')
        keyword_btn4 = ui.button('圖中有幾瓶有藍色瓶蓋的飲料', on_click=lambda: fill_keyword("圖中有幾瓶有藍色瓶蓋的飲料")).classes('bg-blue-500')
        keyword_btn5 = ui.button('圖中有幾瓶包裝上標有 Green Tea', on_click=lambda: fill_keyword("圖中有幾瓶包裝上標有 Green Tea")).classes('bg-blue-500')
        
    ui.label('Click "Start Webcam" to begin streaming').classes('text-sm text-gray-600 mt-2')







# Run the app
ui.run(title='Webcam Display', port=8080, reload=False)



