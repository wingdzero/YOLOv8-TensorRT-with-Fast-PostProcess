import os
import requests
import configparser


config = configparser.ConfigParser()

config.read('configs/config.ini')

# 指定图片目录
image_folder = config.get('Paths', 'input_path')
# image_folder = "images/in"

# 收集所有图片文件
files = []
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 仅限图片格式
        file_path = os.path.join(image_folder, filename)
        files.append(('files', open(file_path, 'rb')))  # 确保使用 'files' 字段

# 发送请求
response = requests.post("http://127.0.0.1:8010/upload/", files=files)


# 打印服务器返回的宽高信息
print(response.json())
