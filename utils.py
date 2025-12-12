import cv2
import base64

def cv_to_base64(img):
    # encode image as JPEG (you can use .png if you prefer)
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("Could not encode image")
    
    # base64 encode
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64