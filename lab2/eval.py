import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(self, model, dataset, device="cpu"if torch.cuda.is_available() else "cuda", output_dir="output"):
        self.model = model.to(device)
        self.data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        self.device = device
        self.output_dir = output_dir

    def evaluate(self):
        self.model.eval()  # 设置模型为评估模式
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in self.data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)  # 前向传播
                predictions = torch.argmax(outputs, dim=1)  # 获取预测类别
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # 转换为 numpy 数组
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # 计算评估指标
        self.compute_metrics(y_true, y_pred)
        self.plot_confusion_matrix(y_true, y_pred)
        self.display_misclassified_samples(y_true, y_pred)

    def compute_metrics(self, y_true, y_pred):
        """ 计算并打印 accuracy, precision, recall, f1-score """
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("=" * 40)

    def plot_confusion_matrix(self, y_true, y_pred):
        """ 绘制混淆矩阵 """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(self.output_dir+'/confusion_matrix.png')

    def display_misclassified_samples(self, y_true, y_pred, num_samples=5):
        misclassified_indices = np.where(y_true != y_pred)[0]
        num_samples = min(num_samples, len(misclassified_indices))
        print(f"Misclassified Samples (showing {num_samples} examples):")

        plt.figure(figsize=(10, 10))
        for i, idx in enumerate(misclassified_indices[:num_samples]):
            image, true_label, predicted_label = self.data_loader.dataset[idx][0], y_true[idx], y_pred[idx]
            image = image.permute(1, 2, 0)
            if image.shape[2] == 1:
                image = image.squeeze(2)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(image, cmap="gray")
            plt.title(f"True: {true_label}\nPred: {predicted_label}")
            plt.axis('off')
            # 打印详细信息
            print(f"Index: {idx}, True Label: {true_label}, Predicted: {predicted_label}")
        plt.savefig(self.output_dir+'/misclassified_samples.png')
        print("=" * 40)


if __name__ == "__main__":
    from MNIST import MNIST
    from model import ModelA, ModelB
    model = ModelA()
    model_name = "ModelA-epochs20-batch64-lr0.0001"
    checkpoint_path = model_name+ '/'+model_name + '_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    test_dataset = MNIST().test_dataset

    evaluator = Evaluator(model=model, dataset=test_dataset, output_dir=model_name)

    evaluator.evaluate()

