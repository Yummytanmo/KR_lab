import os
import torch
from Trans import TransE
from TrainDataLoader import TransDataLoader
from Trainer import Trainer

# 设置参数
in_path = "./WN18/"
batch_size = 128
neg_ent = 1
neg_rel = 0
num_workers = 4
dim = 100
norm = 1
margin = 2.0
alpha = 0.01
learning_rate = 0.001
epochs = 50
save_path = "./WN18_model/"
device = torch.device("cuda")

if __name__ == "__main__":
    # 创建数据加载器
    train_loader = TransDataLoader(in_path=in_path, batch_size=batch_size, neg_ent=neg_ent, neg_rel=neg_rel, num_workers=num_workers, filename="train.txt")
    test_loader = TransDataLoader(in_path=in_path, batch_size=batch_size, neg_ent=neg_ent, neg_rel=neg_rel, num_workers=num_workers, filename="test.txt")
    val_loader = TransDataLoader(in_path=in_path, batch_size=batch_size, neg_ent=neg_ent, neg_rel=neg_rel, num_workers=num_workers, filename="valid.txt")
    # 初始化模型
    ent_num = len(train_loader.dataset.entity2id)
    rel_num = len(train_loader.dataset.relation2id)
    model = TransE(ent_num, rel_num, device, dim=dim, norm=norm, margin=margin, alpha=alpha).to(device)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=epochs,
        save_path=save_path,
        save_steps=10,
        checkpoint_path="./WN18_model/checkpoints/",
        device=device,
    )

    # 开始训练
    trainer.train()

