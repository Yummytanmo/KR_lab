import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

np.random.seed(42)

def compute_distribution(dataset):
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        mean += images.mean([0, 2, 3]) * batch_samples
        std += images.std([0, 2, 3]) * batch_samples
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std


class MNIST():
    def __init__(self):
        self.dataset = {}
        self.dataset['train'] = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        self.dataset['test'] = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        self.mean, self.std = compute_distribution(self.dataset['train'])
        transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),  # [0,1] 归一化
        transforms.Normalize(self.mean, self.std)  # 使用计算得到的均值和标准差
        ])

        self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


    def get_distribution(self):
        return self.mean, self.std
    
    # （2）展示处理后的图像样本（每个类别随机展示至少5个样本）
    def show_image(self):
        import matplotlib.pyplot as plt
        dataset = self.dataset['train']
        
        fig, axes = plt.subplots(10, 5, figsize=(10, 20))
        
        targets = np.array(dataset.targets)
        for digit in range(10):
            indices = np.where(targets == digit)[0]
            chosen = np.random.choice(indices, 5, replace=False)
            for j, idx in enumerate(chosen):
                image, label = dataset[idx]
                unnorm = image * self.std.view(-1, 1, 1) + self.mean.view(-1, 1, 1)
                unnorm = unnorm.squeeze().numpy()
                axes[digit, j].imshow(unnorm, cmap='gray')
                axes[digit, j].set_title(f'Label: {label}')
                axes[digit, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('mnist.png')

        

        

if __name__ == "__main__":
    mnist = MNIST()

    mu, std = mnist.get_distribution()
    print(mu[0].item(), std[0].item())

    # mnist.show_image()