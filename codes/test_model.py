# 导入自定义神经网络库 mynn（你自己实现的框架）
import mynn as nn
import numpy as np
from struct import unpack  # 用于读取二进制数据的格式化解包
import gzip  # 用于解压 MNIST 数据集文件（.gz 格式）
import pickle  # 用于加载保存的模型文件（pickle 格式）

# 实例化一个 MLP 模型对象（多层感知机）
model = nn.models.Model_MLP()

# 从保存的模型文件中加载训练好的权重参数
model.load_model(r'./codes/saved_models/best_model.pickle')

# 测试图片的路径（MNIST 测试集图像，gzip 格式）
test_images_path = r'./codes/dataset/MNIST/t10k-images-idx3-ubyte.gz'

# 测试标签的路径（MNIST 测试集标签，gzip 格式）
test_labels_path = r'./codes/dataset/MNIST/t10k-labels-idx1-ubyte.gz'

# 读取并解压测试图像数据
with gzip.open(test_images_path, 'rb') as f:
    # 读取文件头信息：magic number（文件标识）、图像数量、每张图像的行数和列数
    magic, num, rows, cols = unpack('>4I', f.read(16))
    # 读取图像数据，并转换成 numpy 数组（28x28 像素展平为一维：28*28=784）
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

# 读取并解压测试标签数据
with gzip.open(test_labels_path, 'rb') as f:
    # 读取文件头信息：magic number 和标签数量
    magic, num = unpack('>2I', f.read(8))
    # 读取标签数据，转换成 numpy 数组
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 将像素值归一化到 [0, 1] 区间（原始像素值是 0~255）
test_imgs = test_imgs / 255.0

# 使用模型对测试图像进行预测，得到每个类别的 logits
logits = model(test_imgs)

# 实例化准确率计算工具
acc_fn = nn.metric.Accuracy()

# 调用准确率函数，传入模型的输出结果和真实标签，打印测试集准确率
print("Test Accuracy:", acc_fn(logits, test_labs))
