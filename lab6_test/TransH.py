import torch
import torch.nn as nn
import torch.nn.functional as F

class TransH(nn.Module):
    def __init__(self, ent_num, rel_num, dim=100, margin=1.0, norm=2, device='cpu', C=0.25):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.dim = dim
        self.norm = norm
        self.margin = margin
        self.device = device
        self.C = C  # 约束项权重系数

        # 实体嵌入
        self.ent_emb = nn.Embedding(ent_num, dim)
        # 关系嵌入和法向量
        self.rel_emb = nn.Embedding(rel_num, dim)  # 关系平移向量
        self.rel_norm = nn.Embedding(rel_num, dim)  # 超平面法向量

        # 初始化参数
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_emb.weight.data)
        nn.init.xavier_uniform_(self.rel_norm.weight.data)

        # 归一化实体嵌入
        self.ent_emb.weight.data = F.normalize(self.ent_emb.weight.data, p=2, dim=1)
        self.rel_emb.weight.data = F.normalize(self.rel_emb.weight.data, p=2, dim=1)

        self.criterion = nn.MarginRankingLoss(margin)

    def forward(self, h, r, t):
        # 获取嵌入
        h_emb = self.ent_emb(h)
        t_emb = self.ent_emb(t)
        r_emb = self.rel_emb(r)
        w_r = self.rel_norm(r)

        # 法向量归一化
        w_r = F.normalize(w_r, p=2, dim=1)

        # 关系向量投影到超平面
        r_emb = r_emb - torch.sum(r_emb * w_r, dim=1, keepdim=True) * w_r

        # 实体投影到超平面
        h_proj = h_emb - torch.sum(h_emb * w_r, dim=1, keepdim=True) * w_r
        t_proj = t_emb - torch.sum(t_emb * w_r, dim=1, keepdim=True) * w_r

        return torch.norm(h_proj + r_emb - t_proj, p=self.norm, dim=1)

    def loss(self, pos_dist, neg_dist):
        # 原始损失
        original_loss = self.criterion(pos_dist, neg_dist, torch.tensor([-1], device=self.device))
        
        # 实体约束: Σ max(||e||² - 1, 0)
        ent_norms = torch.norm(self.ent_emb.weight, p=2, dim=1)
        entity_constraint = torch.sum(F.relu(ent_norms ** 2 - 1))
        
        # 关系约束: Σ max[(w·d)²/||d||² - 2, 0]
        rel_w = self.rel_norm.weight
        rel_d = self.rel_emb.weight
        dot_product = torch.sum(rel_w * rel_d, dim=1)
        dr_norm_sq = torch.sum(rel_d ** 2, dim=1)
        term = (dot_product ** 2) / (dr_norm_sq + 1e-8) - 2  # 防止除以零
        rel_constraint = torch.sum(F.relu(term))
        
        # 总损失 = 原始损失 + C*(实体约束 + 关系约束)
        total_loss = original_loss + self.C * (entity_constraint + rel_constraint)
        return total_loss

    def evaluate(self, test_tensor, batch_size=2048, ent_batch=2048, split=False):
        """
        Evaluate link prediction on a test set.
        Returns (hits@10, mean rank) or separate head/tail hits@10.
        """
        self.eval()
        test_tensor = test_tensor.to(self.device)
        total_samples = len(test_tensor) * 2

        all_ranks = []
        hits10_head = 0
        hits10_tail = 0

        with torch.no_grad():
            # Tail prediction
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i : i + batch_size]
                h, r, t_true = batch[:, 0], batch[:, 1], batch[:, 2]

                # Embeddings and normalization vector
                h_emb = self.ent_emb(h)
                r_vec = self.rel_emb(r)
                w_r = F.normalize(self.rel_norm(r), p=2, dim=1)

                # Project embeddings
                r_proj = r_vec - (r_vec * w_r).sum(dim=1, keepdim=True) * w_r
                h_proj = h_emb - (h_emb * w_r).sum(dim=1, keepdim=True) * w_r
                target = h_proj + r_proj  # (B, dim)

                # Score in chunks
                scores = []
                for ent_start in range(0, self.ent_num, ent_batch):
                    ent_end = min(ent_start + ent_batch, self.ent_num)
                    ent_ids = torch.arange(ent_start, ent_end, device=self.device)
                    ent_block = self.ent_emb(ent_ids)
                    ent_proj = ent_block.unsqueeze(0) - (ent_block.unsqueeze(0) * w_r.unsqueeze(1)).sum(
                        dim=-1, keepdim=True) * w_r.unsqueeze(1)

                    chunk_scores = torch.norm(
                        target.unsqueeze(1) - ent_proj,
                        p=self.norm,
                        dim=-1
                    )
                    scores.append(chunk_scores)

                scores = torch.cat(scores, dim=1)  # (B, ent_num)
                # Compute ranks: 1-based
                ranks = (scores.argsort(dim=1) == t_true.unsqueeze(1)).nonzero()[:, 1] + 1
                hits10_tail += (ranks <= 10).sum().item()
                all_ranks.extend(ranks.cpu().tolist())

            # Head prediction
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i : i + batch_size]
                h_true, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

                # Embeddings and normalization vector
                t_emb = self.ent_emb(t)
                r_vec = self.rel_emb(r)
                w_r = F.normalize(self.rel_norm(r), p=2, dim=1)

                # Project embeddings
                r_proj = r_vec - (r_vec * w_r).sum(dim=1, keepdim=True) * w_r
                t_proj = t_emb - (t_emb * w_r).sum(dim=1, keepdim=True) * w_r
                target = t_proj - r_proj  # (B, dim)

                # Score in chunks
                scores = []
                for ent_start in range(0, self.ent_num, ent_batch):
                    ent_end = min(ent_start + ent_batch, self.ent_num)
                    ent_ids = torch.arange(ent_start, ent_end, device=self.device)
                    ent_block = self.ent_emb(ent_ids)
                    ent_proj = ent_block.unsqueeze(0) - (ent_block.unsqueeze(0) * w_r.unsqueeze(1)).sum(
                        dim=-1, keepdim=True) * w_r.unsqueeze(1)

                    chunk_scores = torch.norm(
                        target.unsqueeze(1) - ent_proj,
                        p=self.norm,
                        dim=-1
                    )
                    scores.append(chunk_scores)

                scores = torch.cat(scores, dim=1)
                ranks = (scores.argsort(dim=1) == h_true.unsqueeze(1)).nonzero()[:, 1] + 1
                hits10_head += (ranks <= 10).sum().item()
                all_ranks.extend(ranks.cpu().tolist())

        # Final metrics
        if split:
            return hits10_head / len(test_tensor), hits10_tail / len(test_tensor)
        else:
            mean_rank = sum(all_ranks) / len(all_ranks)
            overall_hits10 = (hits10_head + hits10_tail) / total_samples
            return overall_hits10, mean_rank
