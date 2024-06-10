# encoding:utf-8
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from vgg16_pytorch import vgg16,weights_init
from torchvision.models import VGG16_Weights
from data_transforms import data_transforms
import time
import datetime
import copy
import os
# 在需要打印或处理时间的地方，使用以下函数来格式化时间
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))
start_time = time.time()# 初始化时间

# print("torch_version:",torch.__version__)


# 创建一个MPS设备对象
device = torch.device("mps")
num_epochs = 50  # 训练轮数
batch_size = 2
train_root = '../test'  # 训练集根目录
val_root='../val'
save_dir = './recent/test_from0' #（最近训练模型保存位置） 每个 epoch 结束后保存模型和训练状态
os.makedirs(save_dir, exist_ok=True) #False(默认值)，设为True表示当目录已经存在时,os.makedirs() 函数不会抛出异常,而是静默地跳过目录创建操作
model_path = 'vgg16_for_flower102_50epoch_3.pth'  # 最佳模型保存路径
patience = 5  # 当验证集性能连续 5 个 epoch 没有提升时,停止训练

train_dataset = datasets.ImageFolder(root=train_root, transform=data_transforms['train'] )
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0,shuffle=True)
# 在使用MPS加速时，可能需要将num_workers设置为0，以避免与MPS的不兼容问题。
val_dataset = datasets.ImageFolder(root=val_root, transform=data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
num_classes = len(train_dataset.classes)  # 获取类别数
print("num_classes:",num_classes)

# #获取预训练模型的参数权重
# vgg16_pretrained = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# # 冻结前面的层
# for param in vgg16_pretrained.parameters():
#     param.requires_grad = False
# # 修改最后一层的输出大小
# vgg16_pretrained.fc3 = torch.nn.Linear(4096, num_classes)
# vgg16_pretrained_params = vgg16_pretrained.state_dict()

model = vgg16(num_classes=num_classes)
# model.load_state_dict(vgg16_pretrained_params,strict=False)
# 当 strict=True(默认值) 时,要求模型的每一个参数都能找到对应的预训练权重。如果发现任何参数无法完全匹配,就会抛出 RuntimeError 异常。
# 而当 strict=False 时,即使模型中的某些参数无法完全匹配预训练权重,也不会报错,而是跳过这些参数,只加载能够匹配的参数。这在以下情况下很有用:
# 当您的模型结构与预训练模型略有不同时,比如最后一层的输出类别数不同。这种情况下,将 strict=False 可以避免报错,让您可以成功加载预训练模型的大部分权重。
print(model)
model.to(device)
class_to_idx = train_dataset.class_to_idx
print("class_to_idx:",class_to_idx)

# for batch_idx, (data, index) in enumerate(train_loader):
#     print(f"batch {batch_idx},labels: {index+1}")
#     if batch_idx == 0:  # 只打印第一个批次的labels
#         break

best_acc = 0.0  # 初始化最好的验证集准确率
best_model_wts = copy.deepcopy(model.state_dict())  # 最好模型的权重
val_acc_history = []  # 记录验证集准确率的历史
epoch_no_improve = 0  # 当前连续没有提升的 epoch 数

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)

# # 定义损失函数和优化器,只优化最后一层的参数
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.fc3.parameters(), lr=0.00001)
# optimizer = torch.optim.SGD(model.fc3.parameters(), lr=0.0001, momentum=0.9)

# model.apply(weights_init)
# 检查是否有之前保存的模型
saved_models = [f for f in os.listdir(save_dir) if f.startswith('model_epoch_')]
if saved_models:
    # 找到最新保存的模型
    latest_model = max(saved_models, key=lambda x: int(x.split('_')[-1][:-4]))
    print(f'------------------------------------------------------Found saved model: {latest_model}')
    print(os.path.join(save_dir, latest_model))
    model.load_state_dict(torch.load(os.path.join(save_dir, latest_model)))
    # # 冻结前面的层
    # for param in model.parameters():
    #     param.requires_grad = False
    # # 修改最后一层的输出大小
    # model.fc3 = torch.nn.Linear(4096, num_classes)
    # model_fc3 = model.to(device)
    # vgg16_pretrained_params = model_fc3.state_dict()
    # model.load_state_dict(vgg16_pretrained_params, strict=False)

else:
    print('No saved models found.')

# 训练模型
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):#这种形式将起始索引设置为 0，起始索引默认为 1。
        # inputs, labels = data
        # print("inputs=data[0].shape:",data[0].shape)
        # print("labels=data[1].shape:",data[1].shape)
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("outputs.shape:", outputs.shape)
        # print("outputs:", outputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f'------------------------------------------------------Epoch {epoch + 1},'
          f' Loss: {running_loss / (i + 1):.4f}, Duration: {format_time(epoch_duration)}')

    # 如果不是最后一个 epoch，则预估剩余时间
    if epoch < num_epochs - 1:
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = remaining_epochs * epoch_duration
        print(f'Remain time: {format_time(estimated_remaining_time)}')

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 确保不会计算梯度
        valid_loss = 0.0
        correct = 0
        total = 0
        for data in val_loader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            # print("outputs.shape:",outputs.shape)
            loss = criterion(outputs,labels)
            valid_loss += loss.item()
            _, outputs_index = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print("predicted_labels:",outputs_index+1)
            # print("labels:",labels+1)
            correct += torch.eq(outputs_index+1, labels+1).sum().item()
        print(f'Validation Loss: {(valid_loss / len(val_loader)):.4f}, Accuracy: { (correct / total):.4f}')

        val_acc = 100 * correct / total
        val_acc_history.append(val_acc)
        # 更新最好的模型权重
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= patience:
                print(f'------------------------------------------------------Early stopping at epoch {epoch}')
                break

    # 保存最近的模型和训练状态
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))
    print(f'Model(training state)saved to {save_dir}')
    model.train()  # 设置模型为训练模式

print('------------------------------------------------------Finished Training')
total_time = time.time() - start_time
print(f'Total training time: {format_time(total_time)}')

# 将模型参数恢复到最好的状态
model.load_state_dict(best_model_wts)

torch.save(model.state_dict(), model_path)
print('model saved to ',model_path)