import numpy as np
import os
import csv
import pickle
from tqdm import tqdm
from mynn.op import compute_softmax

# 定义模型训练器 RunnerM 类
class RunnerM():
    def __init__(self, model, optimizer, metric, loss_fn, batch_size, scheduler):
        """
        初始化 RunnerM 对象
        :param model: 需要训练的模型
        :param optimizer: 优化器对象(如 SGD)
        :param metric: 评估指标对象(如 Accuracy)
        :param loss_fn: 损失函数对象
        :param batch_size: 每个 mini-batch 的样本数
        :param scheduler: 学习率调度器（可选）
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric  # 确保是已实例化的评估指标
        self.scheduler = scheduler
        self.batch_size = batch_size

        # 训练过程中记录的指标
        self.train_scores = []  # 每次迭代的训练集得分（准确率）
        self.dev_scores = []    # 每次迭代的验证集得分（准确率）
        self.train_loss = []    # 每次迭代的训练集损失
        self.dev_loss = []      # 每次迭代的验证集损失

        # 用于存储训练日志到 CSV 文件
        self.log_records = []

    def train(self, train_set, dev_set, **kwargs):
        
        num_epochs = kwargs.get("num_epochs", 0)  # 默认训练轮数为 0
        log_iters = kwargs.get("log_iters", 100)  # 日志打印间隔（目前没启用）
        save_dir = kwargs.get("save_dir", "best_model")  # 模型保存目录

        # 如果保存目录不存在，则创建
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0  # 初始化最佳验证集分数
        X_train, y_train = train_set
        total_iterations = int(np.ceil(X_train.shape[0] / self.batch_size))  # 计算每个 epoch 的迭代次数

        # 训练循环
        for epoch in range(num_epochs):
            X, y = train_set
            assert X.shape[0] == y.shape[0]  # 确保输入和标签的样本数一致

            # 每个 epoch 开始前，打乱训练数据顺序（提升泛化能力）
            idx = np.random.permutation(range(X.shape[0]))
            X = X[idx]
            y = y[idx]

            dev_loss = 0.0
            dev_score = 0.0
            epoch_loss = 0.0
            epoch_score = 0.0
            
            # 如果使用了学习率调度器，则更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # tqdm 用于显示训练进度条
            with tqdm(total=total_iterations, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for iteration in range(total_iterations):
                    # 获取当前 batch 的训练数据
                    train_X = X[iteration * self.batch_size: (iteration + 1) * self.batch_size]
                    train_y = y[iteration * self.batch_size: (iteration + 1) * self.batch_size]

                    # 调整输入维度，适配模型输入要求（batch_size, channel, height, width）
                    train_X = train_X.reshape(-1, 1, 28, 28)

                    # 前向传播：计算模型输出 logits
                    logits = self.model(train_X)

                    trn_loss = self.loss_fn(logits, train_y)
                    epoch_loss += trn_loss
                    
                    # Q4: Implement the cross entropy loss.
                    probs = compute_softmax(logits)
                    trn_score = self.metric(probs, train_y)
                    epoch_score += trn_score

                    # 反向传播：计算梯度并更新参数
                    self.loss_fn.backward()
                    self.optimizer.step()

                    # 清空模型中的梯度缓存
                    self.model.clear_grad()

                    # 更新进度条中的显示信息
                    pbar.set_postfix({
                        "train_loss": f"{trn_loss:.4f}",
                        "train_score": f"{trn_score:.4f}"
                    })
                    pbar.update(1)
                    
            
            # 记录每个 epoch 的平均训练损失与准确率
            avg_loss = epoch_loss / total_iterations
            avg_score = epoch_score / total_iterations
            self.train_loss.append(avg_loss)
            self.train_scores.append(avg_score)

            # 在验证集上评估当前模型表现
            dev_score, dev_loss = self.evaluate(dev_set)
            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)

            # 记录当前训练过程的日志信息
            log = {
                "epoch": epoch + 1,
                "iteration": iteration,
                "train_loss": float(trn_loss),
                "train_score": float(trn_score),
                "dev_loss": float(dev_loss),
                "dev_score": float(dev_score)
            }
            self.log_records.append(log)
            if trn_score > best_score:
                save_path = os.path.join(save_dir, 'model.pickle')
                self.save_model(save_path)  # 保存当前最佳模型
                tqdm.write(f"🎉 Best accuracy updated: {best_score:.5f} --> {trn_score:.5f}")
                best_score = trn_score  # 更新最佳得分

        self.best_score = best_score  # 保存训练过程中的最佳分数
        
    def evaluate(self, data_set):
        X, y = data_set
        X = X.reshape(-1, 1, 28, 28)  # 调整输入形状

        logits = self.model(X)  # 前向传播
        loss = self.loss_fn(logits, y)  # 计算损失

        probs = compute_softmax(logits)  # 计算概率分布
        score = self.metric(probs, y)  # 计算准确率

        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)

    def save_log_to_csv(self, file_path):
        if not self.log_records:
            print("No logs to save.")
            return

        keys = self.log_records[0].keys()  # 提取日志中的字段名作为 CSV 表头
        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()  # 写入表头
            writer.writerows(self.log_records)  # 写入所有日志记录


class RunnerM_CNN():
    def __init__(self, model, train_images, train_labels, test_images, test_labels, best_model_path):
        self.model = model
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.best_model_path = best_model_path

        self.train_scores = []  # 每次迭代的训练集得分（准确率）
        self.dev_scores = []    # 每次迭代的验证集得分（准确率）
        self.train_loss = []    # 每次迭代的训练集损失
        self.dev_loss = []      # 每次迭代的验证集损失

        # 用于存储训练日志到 CSV 文件
        self.log_records = []

    def train(self, image, label, lr):
        """单张图像训练"""
        out, loss, acc = self.model.forward(image, label)
        self.model.backward(label, out, lr)
        return loss, acc

    def run_test(self):
        """评估模型在验证集上的表现"""
        total_loss = 0
        total_correct = 0

        for im, label in zip(self.test_images, self.test_labels):
            _, loss, acc = self.model.forward(im, label)
            total_loss += loss
            total_correct += acc

        val_loss = total_loss / len(self.test_images)
        val_acc = total_correct / len(self.test_images)
        return val_loss, val_acc

    def save_model(self):
        folder = os.path.dirname(self.best_model_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(self.best_model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def run_train(self, epochs, lr, batch_size):
        num_batches = int(np.ceil(len(self.train_images) / batch_size))
        previous_acc = 0.0  # 上一个 epoch 的训练准确率

        for epoch in range(epochs):
            permutation = np.random.permutation(len(self.train_images))
            self.train_images = self.train_images[permutation]
            self.train_labels = self.train_labels[permutation]

            total_loss = 0
            total_correct = 0
            
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for i, (im, label) in enumerate(zip(self.train_images, self.train_labels)):
                    loss, correct = self.train(im, label, lr=lr)
                    total_loss += loss
                    total_correct += correct

                    avg_loss = total_loss / (i + 1)
                    avg_acc = total_correct / (i + 1)

                    pbar.set_postfix({
                        'train_loss': f'{avg_loss:.4f}',
                        'train_scores': f'{avg_acc:.4f}'
                    })
                    pbar.update(1)

            # 每个 epoch 的训练结果
            self.train_loss.append(avg_loss)
            self.train_scores.append(avg_acc)

            # 【新增】跑一遍验证集，并记录
            val_loss, val_acc = self.run_test()
            self.dev_loss.append(val_loss)
            self.dev_scores.append(val_acc)

            log = {
                "epoch": epoch + 1,
                "train_loss": float(avg_loss),
                "train_scores": float(avg_acc),
                "val_loss": float(val_loss),
                "val_scores": float(val_acc)
            }
            self.log_records.append(log)

            # 比较并输出精简信息
            if avg_acc > previous_acc:
                print(f"✅ Training accuracy improved: {previous_acc:.4f} --> {avg_acc:.4f}")
                previous_acc = avg_acc
                self.save_model()
