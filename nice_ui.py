import base64
import cv2
from nicegui import ui

video_capture = cv2.VideoCapture(0)
ui_interactive_image = ui.interactive_image()

def lazy_update() -> None:
    global video_capture, ui_interactive_image
    ret, frame = video_capture.read(0)
    if ret and ui_interactive_image is not None:
        _, imencode_image = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(imencode_image)
        base64_image_string = 'data:image/jpg;base64,' + base64_image.decode('ascii')
        ui_interactive_image.source = base64_image_string
        
ui.timer(interval=0.033, callback=lazy_update)
ui.run(reload=False)