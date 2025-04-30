import itertools
import numpy as np
import gzip
from struct import unpack
import pickle
import mynn as nn
from draw_tools.plot import plot
import matplotlib.pyplot as plt
from mynn.op import he_init
import os
import pandas as pd  # To save results as a table

# 固定随机种子
np.random.seed(309)

# 加载 MNIST 数据集
def load_mnist(augmentation=False):
    with gzip.open('./codes/dataset/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    with gzip.open('./codes/dataset/MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    idx = np.random.permutation(np.arange(num))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    if augmentation:
        aug_train_imgs = np.array([nn.op.augment_image(img) for img in train_imgs])
        aug_train_labs = train_labs.copy()  # 标签不变
        combined_imgs = np.concatenate([train_imgs, aug_train_imgs], axis=0) / 255.0
        combined_labs = np.concatenate([train_labs, aug_train_labs], axis=0)
    else:
        combined_imgs = train_imgs / 255.0
        combined_labs = train_labs
    
    valid_imgs = valid_imgs / 255.0
    return combined_imgs, combined_labs, valid_imgs, valid_labs

# 定义参数网格，加入数据增强选项
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'optimizer': ['SGD', 'StandardMomentum', 'MomentGD'],
    'mu': [0.85, 0.9, 0.95],
    'scheduler': ['StepLR', 'MultiStepLR', 'ExponentialLR', 'None'],
    'weight_decay': [0.0, 1e-4, 1e-6],
    'augmentation': [True, False]  # 添加数据增强选项
}

# 所有参数组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# 搜索结果记录
results = []

# 搜索循环
for i, params in enumerate(param_combinations):
    print(f"\n[{i+1}/{len(param_combinations)}] 正在尝试参数组合: {params}")
    lr = params['learning_rate']
    mu = params['mu']
    wd = params['weight_decay']
    opt = params['optimizer']
    sch = params['scheduler']
    augmentation_used = params['augmentation']  # 获取是否使用数据增强
    
    # Load data with augmentation flag
    train_imgs, train_labs, valid_imgs, valid_labs = load_mnist(augmentation=augmentation_used)
    
    model = nn.models.Model_MLP(
        size_list=[784, 256, 128, 10],
        act_func='ReLU',
        lambda_list=[wd]*3,
        initialize_methods=[he_init]*3
    )
    
    # 初始化优化器
    if opt == 'SGD':
        optimizer = nn.optimizer.SGD(lr, model=model, weight_decay=wd)
    elif opt == 'StandardMomentum':
        optimizer = nn.optimizer.StandardMomentum(lr, model=model, mu=mu)
    elif opt == 'MomentGD':
        optimizer = nn.optimizer.MomentGD(lr=lr, model=model, mu=mu, weight_decay=wd)
    else:
        raise ValueError(f"未知优化器: {opt}")
    
    # 初始化学习率调度器
    if sch == 'StepLR':
        scheduler = nn.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    elif sch == 'MultiStepLR':
        scheduler = nn.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 25], gamma=0.8)
    elif sch == 'ExponentialLR':
        scheduler = nn.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    elif sch == 'None':
        scheduler = None
    else:
        raise ValueError(f"未知调度器: {sch}")
    
    # 初始化损失函数和 Runner
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.Accuracy(), loss_fn, batch_size=256, scheduler=scheduler)
    
    save_path = f"./codes/best_models_aug/run_{i}"
    os.makedirs(save_path, exist_ok=True)
    
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=30, log_iters=100, save_dir=save_path)
    val_acc, val_loss = runner.evaluate((valid_imgs, valid_labs))
    print(f"验证准确率: {val_acc:.4f}, 验证损失: {val_loss:.4f}")

    # 保存每个试验结果
    results.append({
        'index': i,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'params': params,
        'augmentation': augmentation_used
    })

# 将结果保存为表格（CSV文件）
results_df = pd.DataFrame(results)
results_df.to_csv("./codes/param_search_results.csv", index=False)

