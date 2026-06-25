import requests
import json
import base64

def get_image_data(file_path: str) -> bytes:
    try:
        with open(file_path, 'rb') as file:
            file.seek(0)
            data = file.read()
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading image file {file_path}: {str(e)}")

raw_data = get_image_data("C:/hongbo/test/cat.jpg")

# 转换为 base64
base64_data = base64.b64encode(raw_data).decode('utf-8')
print(f"Base64 数据大小: {len(base64_data)} 字符")

payload = {
    "prompt": "请帮我描述这张图片",
    "image_data": [
      {
        "data": base64_data,
        "id": 0
      },
    ]
}

url = "http://127.0.0.1:8088/completion"
headers = {"Content-Type": "application/json"}

resp = requests.post(url, json=payload, headers=headers, timeout=60,
                     proxies={"http": None, "https": None})
print(resp.status_code, resp.text)
