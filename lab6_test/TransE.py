import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加中文注释到 TransE 类
class TransE(nn.Module):
    def __init__(self, ent_num, rel_num, dim=100, margin=1.0, norm=2, device='cpu'):
        super().__init__()
        self.ent_num = ent_num  # 实体数量
        self.rel_num = rel_num  # 关系数量
        self.dim = dim  # 嵌入维度
        self.norm = norm  # 范数类型
        self.margin = margin  # 损失函数的边界值
        self.device = device  # 设备类型

        # 初始化嵌入层
        self.ent_emb = nn.Embedding(ent_num, dim)  # 实体嵌入
        self.rel_emb = nn.Embedding(rel_num, dim)  # 关系嵌入
        nn.init.xavier_uniform_(self.ent_emb.weight.data)  # 使用 Xavier 初始化实体嵌入
        nn.init.xavier_uniform_(self.rel_emb.weight.data)  # 使用 Xavier 初始化关系嵌入
        self.ent_emb.weight.data = F.normalize(self.ent_emb.weight.data, p=2, dim=1)  # 对实体嵌入进行归一化
        self.rel_emb.weight.data = F.normalize(self.rel_emb.weight.data, p=2, dim=1)  # 对关系嵌入进行归一化

        self.criterion = nn.MarginRankingLoss(margin)  # 定义边界排名损失函数

    def forward(self, h, r, t):
        h_emb = self.ent_emb(h)  # 获取头实体嵌入
        r_emb = self.rel_emb(r)  # 获取关系嵌入
        t_emb = self.ent_emb(t)  # 获取尾实体嵌入
        return torch.norm(h_emb + r_emb - t_emb, p=self.norm, dim=1)  # 计算三元组的距离

    def loss(self, pos_dist, neg_dist):
        return self.criterion(pos_dist, neg_dist, torch.tensor([-1], device=self.device))  # 计算损失

    def evaluate(self, test_tensor, batch_size=256, split=False):
        self.eval()  # 设置模型为评估模式
        test_tensor = test_tensor.to(self.device)  # 将测试数据移动到设备
        total_samples = len(test_tensor) * 2  # 总样本数

        all_ranks = []  # 存储所有排名
        hits10_head = 0
        hits10_tail = 0
        with torch.no_grad():
            # 尾实体预测
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i:i+batch_size]
                h, r, t_true = batch[:,0], batch[:,1], batch[:,2]

                h_emb = self.ent_emb(h)  # 获取头实体嵌入
                r_emb = self.rel_emb(r)  # 获取关系嵌入
                scores = torch.cdist(h_emb + r_emb, self.ent_emb.weight, p=self.norm)  # 计算分数

                ranks = (scores.argsort(dim=1) == t_true.unsqueeze(1)).nonzero()[:,1] + 1  # 计算排名
                hits10_tail += (ranks <= 10).sum().item()  # 统计命中前10的数量
                all_ranks.extend(ranks.cpu().tolist())  # 添加排名到列表

            # 头实体预测
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i:i+batch_size]
                h_true, r, t = batch[:,0], batch[:,1], batch[:,2]

                t_emb = self.ent_emb(t)  # 获取尾实体嵌入
                r_emb = self.rel_emb(r)  # 获取关系嵌入
                target = t_emb - r_emb  # 目标向量
                scores = torch.cdist(self.ent_emb.weight, target, p=self.norm).t()  # 计算分数

                ranks = (scores.argsort(dim=1) == h_true.unsqueeze(1)).nonzero()[:,1] + 1  # 计算排名
                hits10_head += (ranks <= 10).sum().item()  # 统计命中前10的数量
                all_ranks.extend(ranks.cpu().tolist())  # 添加排名到列表
        if split:
            return hits10_head / len(test_tensor), hits10_tail / len(test_tensor)
        else:
            return (hits10_head + hits10_tail) / total_samples, sum(all_ranks)/len(all_ranks)  # 返回命中率和平均排名