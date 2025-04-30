import numpy as np
import gzip
from struct import unpack
import os
import pickle
import mynn as nn
import matplotlib.pyplot as plt
from draw_tools.plot import plot

def load_imgs(images_path, labels_path):
    """从 MNIST 数据集中加载图像与标签，并划分训练集与验证集"""
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("请确保 MNIST gzip 文件已正确下载并放置在 ./dataset/MNIST 目录下")

    # 读取图像数据
    with gzip.open(images_path, 'rb') as f_img:
        magic, num, rows, cols = unpack('>4I', f_img.read(16))  # 读取文件头信息
        images = np.frombuffer(f_img.read(), dtype=np.uint8).reshape(num, 28, 28)  # 将图像展平

    # 读取标签数据
    with gzip.open(labels_path, 'rb') as f_lab:
        magic, num_labels = unpack('>2I', f_lab.read(8))  # 读取文件头信息
        labels = np.frombuffer(f_lab.read(), dtype=np.uint8)

    # 打乱顺序
    indices = np.random.permutation(num)  # 打乱所有图像

    # 保存索引，以便复现实验
    with open('idx.pickle', 'wb') as f:
        pickle.dump(indices, f)

    images = images[indices]
    labels = labels[indices]

    # 划分训练集和验证集
    test_images = images[:10000]  # 前 10000 张图片作为验证集
    test_labels = labels[:10000]
    train_images = images[10000:]  # 剩余的作为训练集
    train_labels = labels[10000:]

    return train_images, train_labels, test_images, test_labels

# 数据路径
train_images_path = './codes/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './codes/dataset/MNIST/train-labels-idx1-ubyte.gz'

# （1）原数据
train_images, train_labels, test_images, test_labels = load_imgs(train_images_path, train_labels_path)

# （2）数据增强 + 原数据
# train_images, train_labels, test_images, test_labels = load_imgs(train_images_path, train_labels_path)
# # 在归一化前进行数据增强
# aug_train_imgs = np.array([nn.op.augment_image(img) for img in train_images])
# aug_train_imgs = aug_train_imgs.reshape(-1, 28, 28)
# aug_train_labs = train_labels.copy()  # 标签不变
# # 合并原图与增强图
# train_images = np.concatenate([train_images, aug_train_imgs], axis=0)
# train_labels = np.concatenate([train_labels, aug_train_labs], axis=0)

num_filters = 8
num_classes = 10
epochs = 15
lr = 0.001
batch_size = 4096
best_model_path='./codes/best_models/general_model_cnn.pickle'

model = nn.models.Model_CNN(num_filters, num_classes)
runner = nn.runner.RunnerM_CNN(model, train_images, train_labels, test_images, test_labels, best_model_path)
runner.run_train(epochs=epochs, lr=lr, batch_size=batch_size)


print(f"是否使用数据增强: False")

# 训练完成后，打印信息
# print("数据增强方式：平移 + 旋转 + 缩放")
# print(f"是否使用数据增强：True")

val_acc, val_loss = runner.run_test()
print(f"验证准确率：{val_acc:.4f}")
print(f"验证损失：{val_loss:.4f}")


# 绘制训练过程中的 loss 和 accuracy 曲线
_, axes = plt.subplots(1, 2)  # 创建 1 行 2 列的子图
axes.reshape(-1)  # 确保 axes 是一维数组，便于传递给绘图函数
_.set_tight_layout(1)  # 自动调整子图布局
plot(runner, axes)  # 绘图

plt.show()  # 显示图表