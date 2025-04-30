import mynn as nn  # 导入自己实现的神经网络库
from draw_tools.plot import plot  # 导入绘图工具函数
import numpy as np
from struct import unpack  # 用于解析二进制数据
import gzip  # 用于解压 .gz 格式的文件
import matplotlib.pyplot as plt
import pickle  # 用于保存和加载 Python 对象（如索引）
from mynn.op import he_init, xavier_init, uniform_init

np.random.seed(309)

train_images_path = r'./codes/dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./codes/dataset/MNIST/train-labels-idx1-ubyte.gz'


with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))  # 读取文件头信息
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

# 读取训练标签数据
with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))  # 读取文件头信息
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)  # 读取标签

# 从训练集中随机选择 10000 张图片作为验证集
idx = np.random.permutation(np.arange(num))  # 生成随机排列的索引
# 保存索引，便于以后复现实验
with open('idx.pickle', 'wb') as f:
    pickle.dump(idx, f)

# 使用随机索引打乱数据集
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

# 划分训练集和验证集
valid_imgs = train_imgs[:10000]  # 前 10000 张图片作为验证集
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]  # 剩余的作为训练集
train_labs = train_labs[10000:]

# （1）原数据
# train_imgs = train_imgs / train_imgs.max()
# valid_imgs = valid_imgs / valid_imgs.max()

# （2）数据增强 + 原数据
# 在归一化前进行数据增强
aug_train_imgs = np.array([nn.op.augment_image(img) for img in train_imgs])
aug_train_labs = train_labs.copy()  # 标签不变
# 合并原图与增强图
combined_imgs = np.concatenate([train_imgs, aug_train_imgs], axis=0)
combined_labs = np.concatenate([train_labs, aug_train_labs], axis=0)
# 归一化
combined_imgs = combined_imgs / combined_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()
train_imgs = combined_imgs
train_labs = combined_labs

initialize_methods = [he_init, he_init, he_init]  # 每一层指定初始化方法

input_dim = 784
# nHidden = [128]
nHidden = [256, 128]   # Q1: Change the network structure
output_dim = 10
lr = 0.2
mu = 0.9
weight_decay = 0.000001 # Q3: l_2 regularization method 正则化参数

linear_model = nn.models.Model_MLP(
    size_list = [input_dim] + nHidden + [output_dim],
    act_func='ReLU',
    lambda_list=[1e-4, 1e-4, 1e-4],  # 3 个 Linear 层，对应 3 个 lambda
    initialize_methods=initialize_methods
)

# 优化器
# （1）随机梯度下降法
optimizer = nn.optimizer.SGD(lr, model=linear_model, weight_decay=weight_decay)
# （2）标准动量法
# optimizer = nn.optimizer.StandardMomentum(lr, model=linear_model, mu=mu)
# （3）Q3: 基于权重差分的动量
# optimizer = nn.optimizer.MomentGD(lr=lr, model=linear_model, mu=mu, weight_decay=weight_decay)

# 学习率调节器
# （1）阶梯型学习率调度器
# scheduler = nn.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.99)
# （2）多步学习率调度器
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.95)
# （3）指数衰减学习率调度器
# scheduler = nn.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.Accuracy(), loss_fn, batch_size=256, scheduler=None)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=30, log_iters=100, save_dir=r'./codes/best_models')


#print(f"是否使用数据增强: False")

# 训练完成后，打印信息
print("数据增强方式：平移 + 旋转 + 缩放")
print(f"是否使用数据增强：True")

# 获取验证集最后的准确率和损失
val_acc, val_loss = runner.evaluate((valid_imgs, valid_labs))

print(f"验证准确率：{val_acc:.4f}")
print(f"验证损失：{val_loss:.4f}")

# 绘制训练过程中的 loss 和 accuracy 曲线
_, axes = plt.subplots(1, 2)  # 创建 1 行 2 列的子图
axes.reshape(-1)  # 确保 axes 是一维数组，便于传递给绘图函数
_.set_tight_layout(1)  # 自动调整子图布局
plot(runner, axes)  # 绘图

plt.show()  # 显示图表



