# 基于牛津大学flower102数据集的常见花卉识别的设计和实现
## （基于pytorch的vgg16花卉识别）
写在前面：该代码是本地mac运行的，windows用户可以将出现在里面的"mps"改为"cpu"或者"cuda"

### 一、简介
根据拍照得到的花卉图片识别出图片中的花卉种类

### 二、文件结构
**code**
+ vgg16_pytorch.py           利用pytorch写的vgg16模型,注意最后一层没有softmax
+ flower_name.html           手动在官网保存的带有花卉名称的html文件
+ html_data_clean.py         对html文件进行清洗,提取得到每种花对应的数量(保存在per_flower_sum.json),以及按html文件内花卉的顺序作为类标得到的花名对应文件(保存在flower_label_to_name.json)
+ per_flower_sum.json        每种花的数量
+ flower_label_to_name.json  花名与类标一一对应
+ data_transforms.py         数据预处理(构建利用pytorch进行分类任务需要的数据类型,使用transforms进行数据增强)
+ 01_load_data.py            利用从官网下载的matlab文件（setid.mat和imagelabels.mat）实现对数据集(jpg文件(原始花卉图片)或者segmim文件(经过图像分割后的花卉图片))的划分
+ 02_train.py                训练自己的模型(里面的代码包括的训练方式有从0开始训练的、使用预训练的模型微调、以及检查使用最近保存的模型接着训练的)
+ 03_test_evaluate.py        准确率评估
+ 04_result_show.py          结果可视化展示
+ 05_gui.py                  巨简陋的ui界面(选取照片进行花卉识别)
+ xxxxxxxxxx.pth             模型

**[官网](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ )下载的文件 : https://www.robots.ox.ac.uk/~vgg/data/flowers/102/** 
+ jpg 
+ segmim 
+ setid.mat 
+ imagelabels.mat

**以下文件可删除(运行01_load_data.py会自动创建),是按官网划分得到的数据集**
+ test 
+ train 
+ val

**code/recent 也可以删除(运行02_train.py会自动创建),是模型每个epoch会自动保存所创建的**

### 三、调用方式：
下载好官网所需文件后,按顺序执行0102030405py文件即可

### 四、更多：
[ppt讲解](课设讲解.pptx)：课设讲解.pptx
