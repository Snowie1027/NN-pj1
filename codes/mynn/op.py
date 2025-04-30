from abc import abstractmethod  # 导入抽象基类模块
import numpy as np  # 导入 numpy 用于数值计算
import random
from scipy.ndimage import rotate, shift, zoom


# He 初始化（适合 ReLU 激活函数）
def he_init(size):
    fan_in = size[0]  # 输入单元数目
    return np.random.randn(*size) * np.sqrt(2. / fan_in)

# Xavier 初始化（适合 tanh 或 sigmoid 激活函数）
def xavier_init(size):
    fan_in = size[0]
    fan_out = size[1]
    return np.random.randn(*size) * np.sqrt(2. / (fan_in + fan_out))

# 普通的均匀初始化
def uniform_init(size):
    return np.random.uniform(-0.1, 0.1, size)

# 定义所有层的抽象基类
class Layer():
    def __init__(self) -> None:
        self.optimizable = True  # 默认所有层是可优化的

    @abstractmethod
    def forward():
        """ 前向传播方法，所有子类必须实现 """
        pass

    @abstractmethod
    def backward():
        """ 反向传播方法，所有子类必须实现 """
        pass


class Linear(Layer):
    """
    全连接层（Linear Layer）
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        """
        初始化函数
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param initialize_method: 权重初始化方法，默认使用正态分布
        :param weight_decay: 是否使用权重衰减（正则化）
        :param weight_decay_lambda: 权重衰减系数
        """
        super().__init__()
        # 初始化权重：形状为 (输入维度, 输出维度)
        self.W = initialize_method((in_dim, out_dim))
        # 初始化偏置：形状为 (1, 输出维度)
        self.b = np.zeros((1, out_dim))
        # 用于保存梯度的字典
        self.grads = {'W': None, 'b': None}
        # 用于缓存输入数据，供反向传播使用
        self.input = None
        # 将参数也存到字典中，便于统一管理
        self.params = {'W': self.W, 'b': self.b}

        # 权重衰减相关参数
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        """ 使实例对象可以像函数一样调用，实际上调用的是 forward """
        return self.forward(X)

    def forward(self, X):
        """
        前向传播
        :param X: 输入数据，形状 [batch_size, in_dim]
        :return: 输出数据，形状 [batch_size, out_dim]
        """
        # 如果输入数据维度大于 2，说明有多余的维度（比如图像），需要展平
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        self.input = X  # 缓存输入数据，反向传播时用
        return np.dot(X, self.W) + self.b  # 矩阵乘法加偏置，得到输出

    def backward(self, grad):
        """
        反向传播，计算梯度
        :param grad: 来自上一层的梯度，形状 [batch_size, out_dim]
        :return: 传递给前一层的梯度，形状 [batch_size, in_dim]
        """
        batch_size = self.input.shape[0]

        # 计算权重的梯度
        self.grads['W'] = np.dot(self.input.T, grad) / batch_size
        # 计算偏置的梯度
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size

        # 如果启用权重衰减，添加正则项
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        # 计算传递给前一层的梯度
        return np.dot(grad, self.W.T)

    def clear_grad(self):
        """ 清空保存的梯度 """
        self.grads = {'W': None, 'b': None}

class Conv2D(Layer):
    """
    只适用于灰度图像（单通道）2D输入的 3x3 卷积层，用于教学或 MNIST 任务
    """
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        """
        input: shape (H, W) —单张图像
        return: shape (H-2, W-2, num_filters)
        """
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learn_rate):
        """
        d_L_d_out: shape (H-2, W-2, num_filters)
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新权重
        self.filters -= learn_rate * d_L_d_filters
        return None  # 如果接入后还有 Conv 层，就得返回输入梯度


class ReLU(Layer):
    """
    ReLU 激活函数层
    功能：输入大于 0 时保持原值，输入小于等于 0 时输出 0
    """
    def __init__(self):
        super().__init__()
        self.input = None  # 缓存输入数据，供反向传播使用
        self.optimizable = False  # 此层无可优化参数

    def __call__(self, X):
        # 使得对象可被直接调用，等价于 self.forward(X)
        return self.forward(X)

    def forward(self, X):
        # 前向传播：ReLU 激活函数
        self.input = X  # 保存输入，反向传播时使用
        return np.maximum(0, X)

    def backward(self, grads):
        # 反向传播：计算输入梯度
        # 只有输入大于 0 的部分才会有梯度传递
        return grads * (self.input > 0)


class MultiCrossEntropyLoss(Layer):
    """
    多分类交叉熵损失函数（内部集成 softmax 函数）
    """
    def __init__(self, model=None, max_classes=10):
        super().__init__()
        self.model = model  # 需要传入模型引用，用于反向传播
        self.max_classes = max_classes
        self.has_softmax = True  # 默认集成 softmax

        # 缓存变量，供 backward 使用
        self.preds = None  # 保存 softmax 之后的预测值
        self.labels = None  # 保存真实标签

    def __call__(self, predicts, labels):
        # 调用时直接进行前向传播计算
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        # 前向传播：计算 softmax + 交叉熵损失
        self.preds = compute_softmax(predicts) if self.has_softmax else predicts
        self.labels = labels

        batch_size = predicts.shape[0]  # 批量大小
        # 取出对应正确分类的概率
        correct_class_probs = self.preds[np.arange(batch_size), labels]
        correct_class_probs = np.clip(correct_class_probs, 1e-12, 1.0)  # 避免 log(0)
        loss = -np.mean(np.log(correct_class_probs))  # 计算交叉熵损失均值
        return loss

    def backward(self, upstream_grad=1.0):
        # 反向传播：计算 softmax 的梯度，再计算交叉熵梯度
        batch_size = self.preds.shape[0]
        grad = self.preds.copy()
        grad[np.arange(batch_size), self.labels] -= 1  # 正确类别的梯度为 (p - 1)
        grad /= batch_size  # 除以批量大小，取平均

        grad *= upstream_grad  # 乘以上游梯度（链式法则）

        # 通过模型触发反向传播
        self.model.backward(grad)

    def cancel_soft_max(self):
        # 如果模型最后一层已经有 softmax，可以取消内部 softmax
        self.has_softmax = False
        return self



# MaxPool2D 层实现
class MaxPool2D(Layer):
    def __init__(self):
        super().__init__()

    def iterate_regions(self, image):
        '''
        生成 non-overlapping 的 2x2 区域
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        前向传播，返回下采样后的特征图
        '''
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backward(self, d_out):
        '''
        反向传播，将梯度传回最大值位置
        '''
        d_input = np.zeros_like(self.last_input)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_input[i * 2 + i2, j * 2 + j2, f2] = d_out[i, j, f2]
        return d_input



def compute_softmax(X):
    """
    compute_softmax 函数：多分类任务中将 logits 转化为概率分布
    """
    x_max = np.max(X, axis=1, keepdims=True)  # 减去最大值，提升数值稳定性
    x_exp = np.exp(X - x_max)  # 指数运算
    partition = np.sum(x_exp, axis=1, keepdims=True)  # 归一化因子（分母）
    return x_exp / partition  # 归一化得到概率分布



# 定义 Softmax 层，包含权重、偏置、softmax 激活和反向传播
class Softmax(Layer):
    def __init__(self, input_len, nodes):
        super().__init__()
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        self.totals = np.dot(input, self.weights) + self.biases
        self.last_totals = self.totals

        exp = np.exp(self.totals - np.max(self.totals))  # 数值稳定性提升
        self.output = exp / np.sum(exp)
        return self.output

    def backward(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals - np.max(self.last_totals))
            S = np.sum(t_exp)

            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t
            d_L_d_inputs = self.weights @ d_L_d_t

            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)


# Q6: create more training examples(translations, rotations, resizing) 
def augment_image(image):
    """对 28x28 的图像进行简单的数据增强"""
    image = image.reshape(28, 28)
    op = random.choice(['rotate', 'shift', 'zoom'])
    
    if op == 'rotate':
        angle = random.uniform(-15, 15)
        augmented = rotate(image, angle, reshape=False, mode='nearest')
    elif op == 'shift':
        shift_val = [random.uniform(-2, 2), random.uniform(-2, 2)]
        augmented = shift(image, shift_val, mode='nearest')
    elif op == 'zoom':
        factor = random.uniform(0.9, 1.1)
        h, w = image.shape
        zoomed = zoom(image, factor)
        zh, zw = zoomed.shape
        # 裁剪或填充回 28x28
        if factor < 1.0:
            pad_h = (h - zh) // 2
            pad_w = (w - zw) // 2
            augmented = np.pad(zoomed, ((pad_h, h - zh - pad_h), (pad_w, w - zw - pad_w)), mode='constant')
        else:
            start_h = (zh - h) // 2
            start_w = (zw - w) // 2
            augmented = zoomed[start_h:start_h+h, start_w:start_w+w]
    return augmented.flatten()
