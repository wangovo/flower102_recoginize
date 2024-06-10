# encoding:utf-8
from vgg16_pytorch import vgg16,weights_init
import torch
from torch.utils.data import DataLoader
from data_transforms import data_transforms
from torchvision import datasets

device = torch.device("mps")
test_root='../train'
batch_size=2
model = vgg16(num_classes=102)
model_weights = torch.load('./recent/test_from0/model_epoch_6.pth', map_location=device)
model.load_state_dict(model_weights)
model.to(device)

# 评估模型
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)  # 移动图像到MPS设备
            labels = labels.to(device)  # 移动标签到MPS设备
            outputs = model(images)
            _, outputs_index = torch.max(outputs.data, 1)
            # outputs.data是从outputs张量中剥离出来的纯数据部分,不包含任何梯度信息。
            total += labels.size(0)
            correct += torch.eq(outputs_index+1, labels+1).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 加载测试集
testset = datasets.ImageFolder(root=test_root, transform=data_transforms["test"])
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 在测试集上测试模型
test_accuracy = evaluate_model(model, testloader)
print(f'test_accuracy: {test_accuracy:.2f} %')