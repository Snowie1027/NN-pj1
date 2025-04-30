from .op import *
import pickle

class Model_MLP(Layer):
    def __init__(self, size_list=None, act_func=None, lambda_list=None, initialize_methods=None):
        self.size_list = size_list
        self.act_func = act_func
        self.lambda_list = lambda_list
        self.initialize_methods = initialize_methods

        if size_list is not None and act_func is not None:
            self.layers = []  # 存储模型所有层的列表
            for i in range(len(size_list) - 1):
                # 获取该层初始化方法（如果提供了）
                init_fn = None
                if initialize_methods is not None and i < len(initialize_methods):
                    init_fn = initialize_methods[i]

                # 创建线性层，传入初始化方法
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=init_fn)

                # 设置正则化参数
                if lambda_list is not None and i < len(lambda_list):
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                else:
                    layer.weight_decay = False
                    layer.weight_decay_lambda = 0.0

                self.layers.append(layer)

                # 添加激活函数（除最后一层外）
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        raise NotImplementedError("Logistic 激活函数尚未实现。")
                    elif act_func == 'ReLU':
                        self.layers.append(ReLU())
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, '模型尚未初始化。'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.lambda_list = [param.get('lambda', 0.0) for param in param_list[2:]]
        # 重建模型层
        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            params = param_list[i + 2]
            # 加载参数 W 和 b
            layer.W = params['W']
            layer.b = params['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = params['weight_decay']
            layer.weight_decay_lambda = params['lambda']
            self.layers.append(layer)
            # 添加激活函数层
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplementedError("Logistic 激活函数尚未实现。")
                elif self.act_func == 'ReLU':
                    self.layers.append(ReLU())

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            # 只保存具有参数的层（线性层），跳过激活函数层
            if hasattr(layer, 'params') and layer.params:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': getattr(layer, 'weight_decay', False),
                    'lambda': getattr(layer, 'weight_decay_lambda', 0.0),
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def clear_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for param in layer.params.values():
                    if hasattr(param, 'grad'):
                        param.grad = None

# Q5: Implement the Conv2D operator on your own. And modify your Multi-Layer Perceptron into a CNN.
class Model_CNN(Layer):
    def __init__(self, num_filters, num_classes):
        super().__init__()
        self.num_filters = num_filters  # 将 out_channels 改为 num_filters
        self.num_classes = num_classes

        self.conv = Conv2D(num_filters=num_filters)  # 这里使用 num_filters 参数
        self.pool = MaxPool2D()
        self.softmax = Softmax(input_len=13 * 13 * num_filters, nodes=num_classes)

        self.layers = [self.conv, self.pool, self.softmax]

    def __call__(self, x, label=None):
        return self.forward(x, label)

    def forward(self, x, label=None):
        # 输入单张图像 (28, 28)
        x = (x / 255.0) - 0.5
        out = self.conv.forward(x)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)
        if label is not None:
            loss = -np.log(out[label] + 1e-7)
            acc = 1 if np.argmax(out) == label else 0
            return out, loss, acc
        return out

    def backward(self, label, out, lr):
        """
        反向传播并更新权重

        参数:
        - label: 正确标签
        - out: 前向传播得到的 softmax 输出
        - lr: 学习率
        """
        # 交叉熵损失的梯度，softmax 输出的梯度
        grad = np.zeros(10)  # 初始化一个全为0的数组，大小为类别数（假设是10个类别）
        grad[label] = -1 / (out[label] + 1e-7)  # 计算损失函数相对于 softmax 输出的梯度

        # 依次通过层进行反向传播
        grad = self.softmax.backward(grad, lr)  # Softmax 层的反向传播
        grad = self.pool.backward(grad)  # 池化层的反向传播
        grad = self.conv.backward(grad, lr)  # 卷积层的反向传播

    def save_model(self, path):
        params = {
            'out_channels': self.num_channels,
            'num_classes': self.num_classes,
            'conv_W': self.conv.W,
            'conv_b': self.conv.b,
            'softmax_W': self.softmax.W,
            'softmax_b': self.softmax.b
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)

        self.num_channels = params['out_channels']
        self.num_classes = params['num_classes']

        self.conv = Conv2D(out_channels=self.num_channels)
        self.pool = MaxPool2D()
        self.softmax = Softmax(13 * 13 * 8, 10)

        self.conv.W = params['conv_W']
        self.conv.b = params['conv_b']
        self.softmax.W = params['softmax_W']
        self.softmax.b = params['softmax_b']

        self.layers = [self.conv, self.pool, self.softmax]

    def clear_grad(self):
        if hasattr(self.conv, 'grad_W'):
            self.conv.grad_W = None
        if hasattr(self.conv, 'grad_b'):
            self.conv.grad_b = None
        if hasattr(self.softmax, 'grad_W'):
            self.softmax.grad_W = None
        if hasattr(self.softmax, 'grad_b'):
            self.softmax.grad_b = None