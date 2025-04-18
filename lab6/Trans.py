import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, ent_num, rel_num, device, dim=100, norm=1, margin=2.0, alpha=0.01):
        super(TransE, self).__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.dim = dim
        self.norm = norm
        self.margin = margin
        self.alpha = alpha
        
        self.ent_embeddings = nn.Embedding(self.ent_num, self.dim) 
        torch.nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, 1)
        self.rel_embeddings = nn.Embedding(self.rel_num, self.dim) 
        torch.nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, 1)
        # 损失函数
        self.criterion = nn.MarginRankingLoss(margin=self.margin)
    
    def get_ent_resps(self, ent_idx): #[batch]
        return self.ent_embeddings(ent_idx) # [batch, emb]
        
    def distance(self, h_idx, r_idx, t_idx):
        h_embs = self.ent_embeddings(h_idx) # [batch, emb] 
        r_embs = self.rel_embeddings(r_idx) # [batch, emb] 
        t_embs = self.ent_embeddings(t_idx) # [batch, emb] 
        scores = h_embs + r_embs - t_embs
        # norm 是计算 loss 时的正则化项
        norms = (torch.mean(h_embs.norm(p=self.norm, dim=1) - 1.0) + torch.mean(r_embs ** 2) + torch.mean(t_embs.norm(p=self.norm, dim=1) - 1.0)) / 3
        return scores.norm(p=self.norm, dim=1), norms
        
    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.float, device=self.device) 
        return self.criterion(positive_distances, negative_distances, target)
    
    def forward(self, h_idx, r_idx, t_idx, h_idx_neg, r_idx_neg, t_idx_neg):
        positive_distances, norms = self.distance(h_idx, r_idx, t_idx)
        negative_distances, _ = self.distance(h_idx_neg, r_idx_neg, t_idx_neg)
        loss = self.loss(positive_distances, negative_distances)
        return loss
    
    def predict(self, h_idx, r_idx, entities):
        h_idx = h_idx.to(self.device)
        r_idx = r_idx.to(self.device)
        entities = entities.to(self.device)
        h_embs = self.ent_embeddings(h_idx).unsqueeze(1).repeat(1, entities.size(0), 1)
        r_embs = self.rel_embeddings(r_idx).unsqueeze(1).repeat(1, entities.size(0), 1)
        t_embs = self.ent_embeddings(entities).unsqueeze(0).repeat(h_idx.size(0), 1, 1)
        scores = h_embs + r_embs - t_embs
        scores = scores.norm(p=self.norm, dim=2) # [batch, ent_num]
        return scores

class TransH(nn.Module):
    def __init__(self,):
        super(TransH, self).__init__()
        pass


class TransR(nn.Module):
    def __init__(self,):
        super(TransR, self).__init__()
        pass