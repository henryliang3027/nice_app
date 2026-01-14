import cv2
import base64

def image_file_to_base64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def cv_to_base64(img):
    # encode image as JPEG (you can use .png if you prefer)
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("Could not encode image")

    # base64 encode
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def resize_image(img, size):
    """
    Resize image to given size.

    Args:
        img: Input image (numpy array)
        size: Target size as (width, height) tuple or single int for both dimensions

    Returns:
        Resized image
    """
    if isinstance(size, int):
        size = (size, size)

    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)