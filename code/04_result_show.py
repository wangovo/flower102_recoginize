# encoding:utf-8
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import random
import scipy
import re
from vgg16_pytorch import vgg16

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

# 加载模型
model = vgg16(num_classes=102)
model.load_state_dict(torch.load('./recent/test_from0/model_epoch_6.pth'))
model.eval()

# 读取标签与花名的对应关系
with open('flower_label_to_name.json', 'r') as f:
    label_to_name = json.load(f)

# 随机选择四张图片进行预测
img_folder = '../segmim'  # 图片放在这个文件夹下
img_paths = random.sample([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.jpg')], 4)

# 进行预测
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, img_path in enumerate(img_paths):
    image = preprocess_image(img_path)
    with torch.no_grad():
        output = model(image)
    _, output_index = torch.max(output.data, 1)
    ax = axes[i//2, i%2]
    # pytoch模型标签默认从0开始，将预测结果加1，以匹配从1开始的标签
    predicted_label = str(output_index.item() + 1)
    ax.imshow(Image.open(img_path).resize((224, 224)))
    labels = scipy.io.loadmat('../imagelabels.mat')
    labels = np.array(labels['labels'][0])
    img_number = re.search(r'\d+', os.path.basename(img_path)).group()
    print("img_number:",img_number)
    true_label = labels[int(img_number)]


    print("true_label:",true_label)
    print("predicted_label:",predicted_label)
    # 检查预测是否正确
    if str(predicted_label) in label_to_name:
        if str(true_label) in label_to_name:
            title = f' {label_to_name[str(predicted_label)]} ({str(predicted_label)}) | True: {label_to_name[str(true_label)]} ({str(true_label)})'
        else:
            title = f' {label_to_name[str(predicted_label)]} ({str(predicted_label)}) | True: {str(true_label)}'
    else:
        title = f' {str(predicted_label)} | True: {str(true_label)}'
    ax.set_title(title, color='green' if str(predicted_label) == str(true_label) else 'red')
    ax.axis('off')

plt.show()