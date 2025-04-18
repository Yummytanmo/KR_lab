import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类，用于加载训练数据
class TransDataset(Dataset):
    def __init__(self, in_path="./", filename="train.txt", neg_ent=1, neg_rel=0):
        # 初始化数据集，加载文件并设置参数
        self.in_path = in_path
        self.neg_ent = neg_ent
        self.neg_rel = neg_rel
        self.filename = filename

        # 加载三元组、实体和关系数据
        self.triples = self._load_file(os.path.join(in_path, self.filename))
        self.entity2id = {entity: int(eid) for entity, eid in self._load_file(os.path.join(in_path, "entity2id.txt"))}
        self.relation2id = {relation: int(rid) for relation, rid in self._load_file(os.path.join(in_path, "relation2id.txt"))}
        self.entities = list(self.entity2id.values())

    def _load_file(self, filepath):
        # 加载文件并返回数据
        with open(filepath, "r") as f:
            if "train.txt" in filepath:
                # 针对 train.txt 文件，解析每行的 head, tail, relation
                return [line.strip().split() for line in f.readlines() if line.strip()]
            else:
                # 其他文件保持原有逻辑
                return [line.strip().split() for line in f.readlines()]

    def __len__(self):
        # 返回数据集大小
        return len(self.triples)

    def __getitem__(self, idx):
        # 根据索引返回一个样本
        triple = self.triples[idx]
        head, tail, relation = map(lambda x: self.entity2id.get(x) if x in self.entity2id else self.relation2id.get(x), triple)
        negative_sample = self._generate_negative_samples(head, tail, relation)
        # Ensure the negative_samples list has enough elements before accessing indices
        return {
            "h_id": head,
            "t_id": tail,
            "r_id": relation,
            "h_id_neg": negative_sample['h_id'],
            "t_id_neg": negative_sample['t_id'],
            "r_id_neg": negative_sample['r_id'],
        }

    def _generate_negative_samples(self, head, tail, relation):
        # 生成负采样数据
        if np.random.rand() < 0.5:
            neg_head = np.random.choice(len(self.entities))
            neg_sample = {
                "h_id": neg_head,
                "t_id": tail,
                "r_id": relation,
            }
        else:
            neg_tail = np.random.choice(len(self.entities))
            neg_sample = {
                "h_id": head,
                "t_id": neg_tail,
                "r_id": relation,
            }
        return neg_sample

    def get_all_entity_ids(self):
        # 返回所有实体的ID
        return list(self.entity2id.values())

# 数据加载器类，封装 PyTorch 的 DataLoader
class TransDataLoader:
    def __init__(self, in_path="./", batch_size=128, neg_ent=1, neg_rel=0, num_workers=4, filename="train.txt"):
        # 初始化数据加载器
        self.dataset = TransDataset(in_path, filename=filename, neg_ent=neg_ent, neg_rel=neg_rel)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # 自定义批处理函数，用于将样本整理成批次
        positive_samples = {
            "h_id": torch.tensor([item["h_id"] for item in batch]),
            "t_id": torch.tensor([item["t_id"] for item in batch]),
            "r_id": torch.tensor([item["r_id"] for item in batch]),
        }
        negative_samples = {
            "h_id_neg": torch.tensor([item["h_id_neg"] for item in batch]),
            "t_id_neg": torch.tensor([item["t_id_neg"] for item in batch]),
            "r_id_neg": torch.tensor([item["r_id_neg"] for item in batch]),
        }
        return {
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
        }

    def __iter__(self):
        # 返回数据加载器的迭代器
        return iter(self.dataloader)

    def __len__(self):
        # 返回批次数量
        return len(self.dataloader)

    def get_all_entity_ids(self):
        # 返回所有实体的ID
        return self.dataset.get_all_entity_ids()