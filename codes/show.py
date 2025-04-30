import gzip
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
train_images_path = './codes/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = './codes/dataset/MNIST/train-labels-idx1-ubyte.gz'

def load_images(path):
    with gzip.open(path, 'rb') as f:
        f.read(16)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        images = data.reshape(-1, 28, 28)
    return images

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        f.read(8)
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

# 加载数据
images = load_images(train_images_path)
labels = load_labels(train_labels_path)

# 收集每个数字的10个样本
digit_samples = {i: [] for i in range(10)}
for img, label in zip(images, labels):
    if len(digit_samples[label]) < 10:
        digit_samples[label].append(img)
    if all(len(v) == 10 for v in digit_samples.values()):
        break

# 绘图
plt.figure(figsize=(10, 10))
for row in range(10):
    for col in range(10):
        idx = row * 10 + col
        plt.subplot(10, 10, idx + 1)
        plt.imshow(digit_samples[row][col], cmap='gray')
        plt.axis('off')
        
        # 在第一列图像左上角加数字标签
        if col == 0:
            plt.text(-8, 14, str(row), fontsize=12, weight='bold', ha='right', va='center')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
