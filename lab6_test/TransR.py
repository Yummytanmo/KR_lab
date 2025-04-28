import torch
import torch.nn as nn
import torch.nn.functional as F

class TransR(nn.Module):
    def __init__(self, ent_num, rel_num, dim=100, margin=1
                 .0, norm=2, device='cpu'):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.dim = dim
        self.norm = norm
        self.margin = margin
        self.device = device

        # 实体和关系嵌入
        self.ent_emb = nn.Embedding(ent_num, dim)
        self.rel_emb = nn.Embedding(rel_num, dim)
        # 关系投影矩阵
        self.proj_mat = nn.Embedding(rel_num, dim * dim)

        # 初始化参数（关键修改点1：投影矩阵初始化为单位矩阵）
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)
        # 投影矩阵初始化为单位矩阵
        with torch.no_grad():
            identity = torch.eye(dim).flatten()
            self.proj_mat.weight.data = identity.repeat(rel_num, 1)
        
        # 应用约束（关键修改点2：归一化实体和关系嵌入）
        self.ent_emb.weight.data = F.normalize(self.ent_emb.weight.data, p=2, dim=1)
        self.rel_emb.weight.data = F.normalize(self.rel_emb.weight.data, p=2, dim=1)

        self.criterion = nn.MarginRankingLoss(margin)

    def project_entities(self, e_emb, r):
        """投影实体并应用L2归一化"""
        proj = self.proj_mat(r).view(-1, self.dim, self.dim) # (batch, dim, dim)
        e_proj = torch.bmm(e_emb.unsqueeze(1), proj).squeeze(1) # (batch, dim)
        return F.normalize(e_proj, p=2, dim=1)  # 关键修改点3：投影后归一化

    def forward(self, h, r, t):
        # 获取归一化后的实体嵌入
        h_emb = F.normalize(self.ent_emb(h), p=2, dim=1)
        t_emb = F.normalize(self.ent_emb(t), p=2, dim=1)
        r_emb = F.normalize(self.rel_emb(r), p=2, dim=1)
        
        # 投影并归一化
        h_proj = self.project_entities(h_emb, r)
        t_proj = self.project_entities(t_emb, r)
        
        return torch.norm(h_proj + r_emb - t_proj, p=self.norm, dim=1)

    def loss(self, pos_dist, neg_dist):
        return self.criterion(pos_dist, neg_dist, torch.tensor([-1], device=self.device))
    
    def evaluate(self, test_tensor, batch_size=128, ent_batch=500, split=False):
        """
        Evaluate link prediction on a test set using TransR.
        Returns (hits@10, mean rank) or separate (hits@10_head, hits@10_tail) if split=True.
        """
        self.eval()
        test_tensor = test_tensor.to(self.device)
        total_samples = len(test_tensor) * 2

        all_ranks = []
        hits10_head = 0
        hits10_tail = 0

        with torch.no_grad():
            # 尾实体预测
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i:i+batch_size]
                h, r, t_true = batch[:,0], batch[:,1], batch[:,2]

                # 获取实体和关系向量及关系投影矩阵
                h_emb = self.ent_emb(h)                              # (B, dim)
                r_vec = self.rel_emb(r)                              # (B, dim)
                proj = self.proj_mat(r).view(-1, self.dim, self.dim) # (B, dim, dim)

                # 投影头实体
                h_proj = torch.bmm(h_emb.unsqueeze(1), proj).squeeze(1)  # (B, dim)
                target = h_proj + r_vec                                  # (B, dim)

                # 分块计算所有实体的得分
                scores = []
                for start in range(0, self.ent_num, ent_batch):
                    end = min(start + ent_batch, self.ent_num)
                    ent_ids = torch.arange(start, end, device=self.device)
                    ent_block = self.ent_emb(ent_ids)                      # (chunk_size, dim)
                    ent_proj = torch.matmul(ent_block.unsqueeze(0), proj)  # (B, chunk_size, dim)

                    # 计算距离分数
                    chunk_scores = torch.norm(
                        target.unsqueeze(1) - ent_proj,
                        p=self.norm,
                        dim=-1
                    )  # (B, chunk_size)
                    scores.append(chunk_scores)

                scores = torch.cat(scores, dim=1)  # (B, ent_num)
                # 计算真实尾实体排名
                ranks = (scores.argsort(dim=1) == t_true.unsqueeze(1)).nonzero()[:,1] + 1
                hits10_tail += (ranks <= 10).sum().item()
                all_ranks.extend(ranks.cpu().tolist())

            # 头实体预测
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i:i+batch_size]
                h_true, r, t = batch[:,0], batch[:,1], batch[:,2]

                # 获取实体和关系向量及关系投影矩阵
                t_emb = self.ent_emb(t)                              # (B, dim)
                r_vec = self.rel_emb(r)                              # (B, dim)
                proj = self.proj_mat(r).view(-1, self.dim, self.dim) # (B, dim, dim)

                # 投影尾实体
                t_proj = torch.bmm(t_emb.unsqueeze(1), proj).squeeze(1)  # (B, dim)
                target = t_proj - r_vec                                  # (B, dim)

                # 分块计算所有实体的得分
                scores = []
                for start in range(0, self.ent_num, ent_batch):
                    end = min(start + ent_batch, self.ent_num)
                    ent_ids = torch.arange(start, end, device=self.device)
                    ent_block = self.ent_emb(ent_ids)                      # (chunk_size, dim)
                    ent_proj = torch.matmul(ent_block.unsqueeze(0), proj)  # (B, chunk_size, dim)

                    # 计算距离分数
                    chunk_scores = torch.norm(
                        target.unsqueeze(1) - ent_proj,
                        p=self.norm,
                        dim=-1
                    )  # (B, chunk_size)
                    scores.append(chunk_scores)

                scores = torch.cat(scores, dim=1)  # (B, ent_num)
                # 计算真实头实体排名
                ranks = (scores.argsort(dim=1) == h_true.unsqueeze(1)).nonzero()[:,1] + 1
                hits10_head += (ranks <= 10).sum().item()
                all_ranks.extend(ranks.cpu().tolist())

        # 返回指标
        if split:
            return hits10_head / len(test_tensor), hits10_tail / len(test_tensor)
        else:
            mean_rank = sum(all_ranks) / len(all_ranks)
            overall_hits10 = (hits10_head + hits10_tail) / total_samples
            return overall_hits10, mean_rank
