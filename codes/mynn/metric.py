import numpy as np

# 定义一个通用的指标（Metric）基类，所有具体的指标（如准确率、精确率等）都需要继承这个类
class Metric():
    def __init__(self):
        pass  # 初始化函数，这里不需要做额外操作
    
    def __call__(self, logits, targets):
        """
        这个方法应该由子类实现，用于计算具体的评估指标。
        参数：
        - logits: 模型的输出（通常是未经过 softmax 的 logits）
        - targets: 真实标签
        """
        raise NotImplementedError  # 如果子类没有实现该方法，就会抛出这个异常

# 定义准确率（Accuracy）指标
class Accuracy(Metric):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法

    def __call__(self, logits, labels):
        """
        计算准确率：预测正确的样本数量 / 总样本数量
        参数：
        - logits: 模型的输出（未经过 softmax）
        - labels: 真实标签
        """
        # 取出模型输出中预测概率最高的类别作为最终预测类别
        pred_labels = np.argmax(logits, axis=1)
        # 计算预测正确的样本数量
        correct = np.sum(pred_labels == labels)
        # 总样本数量
        total = labels.shape[0]
        # 返回准确率
        return correct / total
