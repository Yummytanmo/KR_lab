import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader,test_loader, optimizer, epochs, checkpoint_path, save_path, save_steps, device, alpha=0.5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_path = save_path
        self.save_steps = save_steps
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.alpha = alpha
        # Initialize the best model and best accuracy
        self.best_model = None
        self.best_acc = 0.0
        # Convert the list to a NumPy array before using torch.from_numpy
        self.entities = torch.from_numpy(np.array(train_loader.get_all_entity_ids())).to(self.device)

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        positive = data['positive_samples']
        negative = data['negative_samples']
        loss = self.model(
            positive['h_id'].to(self.device),
            positive['r_id'].to(self.device),
            positive['t_id'].to(self.device),
            negative['h_id_neg'].to(self.device),
            negative['r_id_neg'].to(self.device),
            negative['t_id_neg'].to(self.device),
        )
        loss.backward()
        self.optimizer.step()
        # # 要求所有实体和关系嵌入在更新后都保持单位范数，否则随着训练继续，向量长度会偏离，导致距离函数失效
        # with torch.no_grad():
        #     self.model.ent_embeddings.weight.data = F.normalize(
        #         self.model.ent_embeddings.weight.data, p=2, dim=1
        #     )
        #     self.model.rel_embeddings.weight.data = F.normalize(
        #         self.model.rel_embeddings.weight.data, p=2, dim=1
        #     )

        return loss.item()

    def train(self):
        self.model.train()
        epoch_range = tqdm(
            range(self.epochs),
            desc="Epochs",
            unit="epoch"
        )

        for epoch in epoch_range:
            total_loss = 0.0

            # 每个 epoch 内部新建批次进度条
            train_range = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs} Batches",
                unit="batch",
                total=len(self.train_loader),
                leave=False
            )

            for step, data in enumerate(train_range):
                # show data
                print(data)
                loss = self.train_one_step(data)
                total_loss += loss

                # 只更新 postfix，显示当前 loss
                train_range.set_postfix(loss=f"{loss:.4f}")
                train_range.refresh()

            avg_loss = total_loss / len(self.train_loader)
            # 更新外层进度条的 postfix，显示平均 loss
            epoch_range.set_postfix(avg_loss=f"{avg_loss:.4f}")
            epoch_range.refresh()

            if epoch % self.save_steps == 0:
                hits, mean_rank = self.evaluate(self.val_loader)
                print(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Avg Loss: {avg_loss:.4f} - "
                    f"Hits@10: {hits:.4f} - "
                    f"Mean Rank: {mean_rank:.4f}"
                )
                if hits > self.best_acc:
                    self.best_acc = hits
                    self.best_model = copy.deepcopy(self.model)
                    print(f"Model saved at epoch {epoch+1} with Hits@10: {hits:.4f}")
                checkpoint_file = os.path.join(
                    self.checkpoint_path,
                    f"checkpoint_epoch_{epoch+1}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_file)
                print(f"Checkpoint saved at epoch {epoch+1}")

        print("Training completed.")
        # 保存最终模型
        final_file = os.path.join(self.save_path, "final_model.pth")
        torch.save(self.model.state_dict(), final_file)
        print("Final model saved.")
        # 最终评估
        hits, mean_rank = self.evaluate(self.val_loader)
        print(
            f"Final model evaluation - "
            f"Hits@10: {hits:.4f} - "
            f"Mean Rank: {mean_rank:.4f}"
        )
        # 保存最优模型
        if self.best_model is not None:
            best_file = os.path.join(self.save_path, "best_model.pth")
            torch.save(self.best_model.state_dict(), best_file)
            print("Best model saved.")
            hits, mean_rank = self.evaluate(self.val_loader)
            print(
                f"Best model evaluation - "
                f"Hits@10: {hits:.4f} - "
                f"Mean Rank: {mean_rank:.4f}"
            )
        else:
            print("No best model found during training.")



    def evaluate(self, data_loader):
        print("Evaluating...")
        self.model.eval()
        correct = 0
        total = 0
        rank = []
        with torch.no_grad():
            for step, data in enumerate(data_loader):
                # 使用所有实体替换tail，预测得分，并获得实际tail的得分rank
                positive = data['positive_samples'] # batch
                head = positive['h_id']
                relation = positive['r_id']
                
                scores = self.model.predict(
                    head, relation, tail
                )

                # scores为[batch, ent_num]，得到实际tail实体的score rank
                tail = positive['t_id']
                tail = tail.to(self.device)
                tail_scores = scores[torch.arange(scores.size(0)), tail]
                _, indices = torch.sort(scores, dim=1, descending=True)
                ranks = torch.nonzero(indices == tail.view(-1, 1), as_tuple=False)[:, 1]
                ranks = ranks.float() + 1
                rank.append(ranks)
                total += tail.size(0)
                correct += torch.sum(ranks <= 10).item()
        rank = torch.cat(rank, dim=0)
        # Hits@10
        hits = correct / total
        # Mean Rank
        mean_rank = torch.mean(rank).item()
        return hits, mean_rank

