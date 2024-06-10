import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import torch
import json
from vgg16_pytorch import vgg16

# 加载模型
model = vgg16(num_classes=102)
model.load_state_dict(torch.load('vgg16_for_flower102_50epoch_2.pth'))
model.eval()

# 读取标签与花名的对应关系
with open('flower_label_to_name.json', 'r') as f:
    label_to_name = json.load(f)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image).unsqueeze(0)
    image = image / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    return image

def predict_flower(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    _, output_index = torch.max(output.data, 1)
    predicted_label = str(output_index.item() + 1)
    if predicted_label in label_to_name:
        return predicted_label, label_to_name[predicted_label]
    else:
        return predicted_label, f"Label {predicted_label}"

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((224, 224))
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo
        prediction = predict_flower(file_path)
        result_label.config(text=f"Predicted Flower: {prediction}")
def center_window(root, width=600, height=600):
    # 获取屏幕尺寸
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 计算窗口位置
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # 设置窗口大小和位置
    root.geometry(f"{width}x{height}+{x}+{y}")

def on_closing():
    root.destroy()

root = tk.Tk()
root.title("Flower Identification")

# 将窗口放在屏幕正中间
center_window(root, 600, 600)

# 创建GUI元素
image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, font=("Arial", 16))
result_label.pack(pady=20)

select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

close_button = tk.Button(root, text="Close", command=on_closing)
close_button.pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()