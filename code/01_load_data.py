# encoding:utf-8
import scipy.io
import numpy as np
import os
import cv2

labels = scipy.io.loadmat('../imagelabels.mat')
labels = np.array(labels['labels'][0])
print("labels:", labels)
print("label(min,max):",labels.min(),labels.max())

setid = scipy.io.loadmat('../setid.mat')
validation = np.array(setid['valid'][0])
np.random.shuffle(validation)
train = np.array(setid['trnid'][0])
np.random.shuffle(train)
test = np.array(setid['tstid'][0])
np.random.shuffle(test)
print("tain",train)
print("validation:",validation)
print("test:",test)
print("validation(min,max):",validation.min(),validation.max())
print("tain(min,max):",train.min(),train.max())
print("test(min,max):",test.min(),test.max())

flower_dir = list()
for img in os.listdir('../jpg'):
    flower_dir.append(os.path.join("../jpg", img))
flower_dir.sort()
print("flower_dir:",flower_dir) # （已排序）所有花的文件名

segment_dir=list()
for img in os.listdir('../segmim'):
    segment_dir.append(os.path.join("../segmim", img))
segment_dir.sort()
print("segment_dir:",segment_dir) # （已排序）所有花的蒙版的文件名

folder_train="../train"
for tid in train:
    #打开图片并获取标签
    tid = tid-1 # train里面每个值 是直接存放的打乱的图片编号，而不是列表对应索引，所以减一
    path_img = flower_dir[tid]
    path_sgmt = segment_dir[tid]
    lable = labels[tid]
    # print("path_img:", path_img)
    # print("path_sgmt:", path_sgmt)
    # print("label:",lable)

    # img1 = cv2.imread(path_img)
    img2 = cv2.imread(path_sgmt)
    # img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    # image_name = os.path.basename(path_img)
    # print("image_name:", image_name)
    segmim_name = os.path.basename(path_sgmt)
    # print("segmim_name:", segmim_name)

    classes = "class_" + str(lable).zfill(5)  # 使用 zfill 方法将整数转换为五位数字的字符串，并用0补齐
    class_path = os.path.join(folder_train, classes)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    # print("class_path:", class_path)

    # complete_image_path = os.path.join(class_path, image_name)
    # img1.save(complete_image_path)
    # cv2.imwrite(complete_image_path, img1)
    # print("complete_image_path:", complete_image_path)

    complete_segmim_path = os.path.join(class_path, segmim_name)
    # img2.save(complete_segmim_path)
    # cv2.imwrite(complete_segmim_path, img2)
    if not os.path.exists(complete_segmim_path):
        cv2.imwrite(complete_segmim_path, img2)
    else:
        print(f"File {complete_segmim_path} already exists, skipping.")
    # print("complete_segmim_path:", complete_segmim_path)


folder_validation = "../val"
for tid in validation:
    # 打开图片并获取标签
    tid = tid - 1  # train里面每个值 是直接存放的打乱的图片编号，而不是列表对应索引，所以减一
    path_img = flower_dir[tid]
    path_sgmt = segment_dir[tid]
    lable = labels[tid]
    # print("path_img:", path_img)
    # print("path_sgmt:", path_sgmt)
    # print("label:", lable)

    # img1 = cv2.imread(path_img)
    img2 = cv2.imread(path_sgmt)
    # img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    image_name = os.path.basename(path_img)
    # print("image_name:", image_name)
    segmim_name = os.path.basename(path_sgmt)
    # print("segmim_name:", segmim_name)

    classes = "class_" + str(lable).zfill(5)  # 使用 zfill 方法将整数转换为五位数字的字符串，并用0补齐
    class_path = os.path.join(folder_validation, classes)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    # print("class_path:", class_path)

    # complete_image_path = os.path.join(class_path, image_name)
    # img1.save(complete_image_path)
    # cv2.imwrite(complete_image_path, img1)
    # print("complete_image_path:", complete_image_path)

    complete_segmim_path = os.path.join(class_path, segmim_name)
    # img2.save(complete_segmim_path)
    # cv2.imwrite(complete_segmim_path, img2)
    if not os.path.exists(complete_segmim_path):
        cv2.imwrite(complete_segmim_path, img2)
    else:
        print(f"File {complete_segmim_path} already exists, skipping.")
    # print("complete_segmim_path:", complete_segmim_path)

folder_test ="../test"
for tid in test:
    # 打开图片并获取标签
    tid = tid - 1  # train里面每个值 是直接存放的打乱的图片编号，而不是列表对应索引，所以减一
    path_img = flower_dir[tid]
    path_sgmt = segment_dir[tid]
    lable = labels[tid]
    # print("path_img:", path_img)
    # print("path_sgmt:", path_sgmt)
    # print("label:", lable)

    # img1 = cv2.imread(path_img)
    img2 = cv2.imread(path_sgmt)
    # img1 = cv2.resize(img1, (256, 256))
    if img2 is None:
        print("Image not loaded.")
        print("path_sgmt:", path_sgmt)
    else:
        img2 = cv2.resize(img2, (256, 256))
    image_name = os.path.basename(path_img)
    # print("image_name:", image_name)
    segmim_name = os.path.basename(path_sgmt)
    # print("segmim_name:", segmim_name)

    classes = "class_" + str(lable).zfill(5)  # 使用 zfill 方法将整数转换为五位数字的字符串，并用0补齐
    class_path = os.path.join(folder_test, classes)
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    # print("class_path:", class_path)

    # complete_image_path = os.path.join(class_path, image_name)
    # img1.save(complete_image_path)
    # cv2.imwrite(complete_image_path, img1)
    # print("complete_image_path:", complete_image_path)

    complete_segmim_path = os.path.join(class_path, segmim_name)
    # img2.save(complete_segmim_path)
    # cv2.imwrite(complete_segmim_path, img2)
    if not os.path.exists(complete_segmim_path):
        cv2.imwrite(complete_segmim_path, img2)
    else:
        print(f"File {complete_segmim_path} already exists, skipping.")
    # print("complete_segmim_path:", complete_segmim_path)