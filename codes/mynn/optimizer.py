from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, lr, model) -> None:
        self.lr = lr         # 初始学习率，仅作参考
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, lr, model, weight_decay=0.0, max_grad_norm=None):
        super().__init__(lr, model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def step(self):
        for layer in self.model.layers:
            if getattr(layer, 'optimizable', False):
                grads = layer.grads

                # 梯度裁剪
                if self.max_grad_norm is not None:
                    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
                    if total_norm > self.max_grad_norm:
                        scale = self.max_grad_norm / (total_norm + 1e-6)
                        for key in grads:
                            grads[key] *= scale

                for key in layer.params:
                    grad = grads[key]
                    if self.weight_decay > 0:
                        grad = grad + self.weight_decay * layer.params[key]

                    layer.params[key] -= self.lr * grad


class StandardMomentum(Optimizer):
    def __init__(self, lr, model, mu, weight_decay=0.0):
        super().__init__(lr, model)
        self.lr = lr
        self.mu = mu
        self.weight_decay = weight_decay
        self.velocity = {}
        for idx, layer in enumerate(self.model.layers):
            if getattr(layer, 'optimizable', False):
                self.velocity[idx] = {key: np.zeros_like(value) for key, value in layer.params.items()}

    def step(self):
        for idx, layer in enumerate(self.model.layers):
            if getattr(layer, 'optimizable', False):
                for key in layer.params.keys():
                    grad = layer.grads[key].copy()  # 注意copy，保护原梯度

                    # 加上L2正则化
                    if self.weight_decay > 0:
                        grad += self.weight_decay * layer.params[key]

                    # 标准Momentum更新
                    self.velocity[idx][key] = self.mu * self.velocity[idx][key] + grad
                    layer.params[key] -= self.lr * self.velocity[idx][key]  # ✅ 替换



class MomentGD(Optimizer):
    def __init__(self, lr, model, mu, weight_decay, lr_schedule=None):
        super().__init__(lr, model)
        self.mu = mu
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule  # 存储学习率调度器
        self.velocity = {}
        self.prev_params = {}

        # 初始化每一层的动量和权重
        for idx, layer in enumerate(self.model.layers):
            if getattr(layer, 'optimizable', False):
                self.velocity[idx] = {key: np.zeros_like(value) for key, value in layer.params.items()}
                self.prev_params[idx] = {key: np.copy(value) for key, value in layer.params.items()}
    def step(self):
        """
        执行优化器的一步（更新权重）
        """
        # 如果有学习率调度器，更新学习率
        if self.lr_schedule is not None:
            self.lr = self.lr_schedule.get_lr(self.lr)  # 获取当前学习率

        # 遍历每一层
        for idx, layer in enumerate(self.model.layers):
            if getattr(layer, 'optimizable', False):
                for key in layer.params.keys():
                    grad = layer.grads[key].copy()

                    # 加入L2正则化
                    if self.weight_decay > 0:
                        grad += self.weight_decay * layer.params[key]

                    # 使用更新后的学习率来更新参数
                    layer.params[key] -= self.lr * self.velocity[idx][key] + self.mu * (layer.params[key] - self.prev_params[idx][key])

                    # 动量更新公式
                    self.velocity[idx][key] = self.mu * self.velocity[idx][key] + grad
                    layer.params[key] -= self.lr * self.velocity[idx][key] + self.mu * (layer.params[key] - self.prev_params[idx][key])

                    # 更新上一轮的权重
                    self.prev_params[idx][key] = np.copy(layer.params[key])
