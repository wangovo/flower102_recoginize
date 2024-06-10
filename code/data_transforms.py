# encoding:utf-8
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

#路径设置
data_dir = '../' # 当前文件夹下的flowerdata目录
batch_size = 2
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/val'
# test_dir = data_dir + '/test'

data_transforms = {
    'train': transforms.Compose([
                                transforms.RandomRotation(45), # 随机旋转 -45度到45度之间
                                 transforms.CenterCrop(224), # 从中心处开始裁剪
                                 transforms.RandomHorizontalFlip(p = 0.5), # 随机水平翻转 # 以某个随机的概率决定是否翻转 55开
                                 transforms.RandomVerticalFlip(p = 0.5), # 随机垂直翻转 # 以某个随机的概率决定是否翻转 55开
                                 transforms.ColorJitter(brightness = 0.2, contrast = 0.1, saturation = 0.1, hue = 0.1),# 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                 transforms.RandomGrayscale(p = 0.025), # 以概率转换为灰度图，三通道RGB # 灰度图转换以后也是三个通道，但是只是RGB是一样的 增加一些随机的灰度图像
                                 transforms.ToTensor(),#将 PIL 图像或者 NumPy 数组转换为 PyTorch 中的张量（Tensor）
                                 # 转换数据类型：将图像的像素值从int型（uint8） 从0-255转换到 [0, 1] 范围内浮点类型。
                                 # 调整维度：将图像的维度从 (H, W, C) 调整为 (C, H, W)。
                                 # 这里的 H、W 和 C 分别代表图像的高度、宽度和颜色通道数。
                                 # 例如，一个形状为 (224, 224, 3) 的 PIL 图像（即高度和宽度都是 224 像素，有 3 个颜色通道的 RGB 图像）
                                 # 在被 ToTensor() 转换后，会变成一个形状为 (3, 224, 224) 的 PyTorch 张量。
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # 均值，标准差

                                ]),
    'val': transforms.Compose([
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
    'test':transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                # transforms.RandomRotation(45),  # 随机旋转 -45度到45度之间
                                # transforms.CenterCrop(224),  # 从中心处开始裁剪
                                # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 # 以某个随机的概率决定是否翻转 55开
                                # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转 # 以某个随机的概率决定是否翻转 55开
                                # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                # transforms.RandomGrayscale(p=0.025),  # 以概率转换为灰度图，三通道RGB # 灰度图转换以后也是三个通道，但是只是RGB是一样的 增加一些随机的灰度图像
                                # transforms.ToTensor(),  # 将 PIL 图像或者 NumPy 数组转换为 PyTorch 中的张量（Tensor）
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train', 'val','test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True,num_workers=0,) for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
class_names = image_datasets['train'].classes

print("dataset_sizes:",dataset_sizes)
print("class_names:",class_names)