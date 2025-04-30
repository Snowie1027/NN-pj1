import matplotlib.pyplot as plt

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]

    # 以迭代次数为基础生成 X 轴数据
    iterations = [i for i in range(len(runner.train_scores))]

    # 绘制训练损失变化曲线
    axes[0].plot(iterations, runner.train_loss, color=train_color, label="Train loss")
    # 绘制评价损失变化曲线
    axes[0].plot(iterations, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
    # 设置坐标轴标签和图例
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_title("Loss Curve")
    axes[0].legend(loc='upper right')

    # 绘制训练准确率变化曲线
    axes[1].plot(iterations, runner.train_scores, color=train_color, label="Train Accuracy")
    # 绘制评价准确率变化曲线
    axes[1].plot(iterations, runner.dev_scores, color=dev_color, linestyle="--", label="Dev Accuracy")
    # 设置坐标轴标签和图例
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend(loc='lower right')

    # 调整布局，避免图形重叠
    plt.tight_layout()
