import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip

def load_first_mnist_image(image_path, label_path):
    with gzip.open(image_path, 'rb') as f_img:
        _ = int.from_bytes(f_img.read(4), 'big')  # magic number
        _ = int.from_bytes(f_img.read(4), 'big')  # num images
        rows = int.from_bytes(f_img.read(4), 'big')
        cols = int.from_bytes(f_img.read(4), 'big')
        buf = f_img.read(rows * cols)  # 只读取一张
        image = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols) / 255.0  # 归一化

    with gzip.open(label_path, 'rb') as f_lbl:
        _ = int.from_bytes(f_lbl.read(4), 'big')  # magic number
        _ = int.from_bytes(f_lbl.read(4), 'big')  # num labels
        label = int.from_bytes(f_lbl.read(1), 'big')  # 第一张图的标签

    return image, label


def visualize_filters(model_path, num_filters=8):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    filters = model.conv.filters  # (num_filters, 3, 3)
    
    num_rows = 2
    num_cols = (num_filters + 1) // 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()

    for idx in range(num_filters):
        ax = axes[idx]
        ax.imshow(filters[idx], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {idx}')
    
    for idx in range(num_filters, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_feature_maps(model_path, sample_image):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    feature_maps = model.conv.forward(sample_image)  # 输出 shape: (H-2, W-2, num_filters)
    num_filters = feature_maps.shape[-1]

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    axes = axes.flatten()

    for idx in range(min(num_filters, 8)):
        ax = axes[idx]
        fmap = feature_maps[:, :, idx]
        fmap = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap) + 1e-5)
        ax.imshow(fmap, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Feature {idx}')

    plt.tight_layout()
    plt.show()


def predict(model, sample_image):
    output = model.forward(sample_image)
    exp_output = np.exp(output - np.max(output))
    probabilities = exp_output / np.sum(exp_output)
    return np.argmax(probabilities)

def plot_all_in_one(sample_image, label, filters, feature_maps, num_filters=8):
    import matplotlib.pyplot as plt

    num_cols = max(num_filters, 8)
    fig, axes = plt.subplots(3, num_cols, figsize=(num_cols * 2, 6))

    # 第1行：原始图像
    for i in range(num_cols):
        axes[0, i].axis('off')
    axes[0, 0].imshow(sample_image, cmap='gray')
    axes[0, 0].set_title(f'Input (Label: {label})')

    # 第2行：卷积核
    for i in range(num_filters):
        axes[1, i].imshow(filters[i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Filter {i}')
    for i in range(num_filters, num_cols):
        axes[1, i].axis('off')

    # 第3行：特征图
    for i in range(min(num_filters, feature_maps.shape[-1])):
        fmap = feature_maps[:, :, i]
        fmap = (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap) + 1e-5)
        axes[2, i].imshow(fmap, cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title(f'Feature {i}')
    for i in range(num_filters, num_cols):
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Start running...")
    model_path = './codes/best_models/general_model_cnn.pickle'
    image_path = './codes/dataset/MNIST/train-images-idx3-ubyte.gz'
    label_path = './codes/dataset/MNIST/train-labels-idx1-ubyte.gz'

    sample_image, label = load_first_mnist_image(image_path, label_path)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    filters = model.conv.filters
    feature_maps = model.conv.forward(sample_image)

    plot_all_in_one(sample_image, label, filters, feature_maps, num_filters=8)
