from ollama import chat
# from pathlib import Path

# Pass in the path to the image
# path = input('Please enter the path to the image: ')

# You can also pass in base64 encoded image data
# img = base64.b64encode(Path(path).read_bytes()).decode()
# or the raw bytes
# img = Path(path).read_bytes()

response = chat(
  model='ministral-3:3b',
  messages=[
    {
      'role': 'user',
      'content': '統計圖中的商品，簡單回答商品名稱和數量，回答格式：名稱:數量',
      'images': ['/home/b40351/Downloads/20251210_105454.jpg'],
    }
  ],
)

print(response.message.content)