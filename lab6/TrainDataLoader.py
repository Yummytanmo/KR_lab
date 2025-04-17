import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类，用于加载训练数据
class TrainDataset(Dataset):
    def __init__(self, in_path="./", neg_ent=1, neg_rel=0):
        # 初始化数据集，加载文件并设置参数
        self.in_path = in_path
        self.neg_ent = neg_ent
        self.neg_rel = neg_rel

        # 加载三元组、实体和关系数据
        self.triples = self._load_file(os.path.join(in_path, "train.txt"))
        self.entity2id = {entity: int(eid) for entity, eid in self._load_file(os.path.join(in_path, "entity2id.txt"))}
        self.relation2id = {relation: int(rid) for relation, rid in self._load_file(os.path.join(in_path, "relation2id.txt"))}

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
        head, tail, relation = map(lambda x: self.entity2id.get(x, -1) if x in self.entity2id else self.relation2id.get(x, -1), triple)
        negative_samples = self._generate_negative_samples(head, tail, relation)
        return {
            "head": head,
            "tail": tail,
            "relation": relation,
            "negative_samples": negative_samples,
        }

    def _generate_negative_samples(self, head, tail, relation):
        # 生成负采样数据
        neg_samples = []
        for _ in range(self.neg_ent):
            neg_head = np.random.choice(len(self.entities))
            neg_samples.append((neg_head, tail, relation))
        for _ in range(self.neg_rel):
            neg_rel = np.random.choice(len(self.relations))
            neg_samples.append((head, tail, neg_rel))
        return neg_samples

# 数据加载器类，封装 PyTorch 的 DataLoader
class TrainDataLoader:
    def __init__(self, in_path="./", batch_size=128, neg_ent=1, neg_rel=0, num_workers=4):
        # 初始化数据加载器
        self.dataset = TrainDataset(in_path, neg_ent, neg_rel)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        # 自定义批处理函数，用于将样本整理成批次
        heads = torch.tensor([item["head"] for item in batch], dtype=torch.long)
        tails = torch.tensor([item["tail"] for item in batch], dtype=torch.long)
        relations = torch.tensor([item["relation"] for item in batch], dtype=torch.long)
        negative_samples = [item["negative_samples"] for item in batch]
        return {
            "heads": heads,
            "tails": tails,
            "relations": relations,
            "negative_samples": negative_samples,
        }

    def __iter__(self):
        # 返回数据加载器的迭代器
        return iter(self.dataloader)

    def __len__(self):
        # 返回批次数量
        return len(self.dataloader)