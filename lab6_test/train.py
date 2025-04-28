# train.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import load_id_triples, load_mapping, load_triples, KGDataset, collate_fn
from TransE import TransE
from TransH import TransH
from TransR import TransR 
from tqdm import tqdm

# 配置参数
config = {
    'data_dir': './WN18',
    'n_epoch': 500,
    'batch_size': 10240,
    'dim': 50,
    'margin': 4.0,
    'norm': 1,  # 1或2范数
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
output_dir =  config['data_dir'] +  f"Transr_{config['n_epoch']}_{config['dim']}_{config['margin']}_{config['norm']}_{config['lr']}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 加载数据
ent_map = load_mapping(f"{config['data_dir']}/entity2id.txt")
rel_map = load_mapping(f"{config['data_dir']}/relation2id.txt")
train_triples = load_triples(f"{config['data_dir']}/train.txt", ent_map, rel_map)
valid_triples = load_triples(f"{config['data_dir']}/valid.txt", ent_map, rel_map)

# 创建模型
# model = TransE(
#     ent_num=len(ent_map),
#     rel_num=len(rel_map),
#     dim=config['dim'],
#     margin=config['margin'],
#     norm=config['norm'],
#     device=config['device']
# ).to(config['device'])
# model = TransH(
#     ent_num=len(ent_map),
#     rel_num=len(rel_map),
#     dim=config['dim'],
#     margin=config['margin'],
#     norm=config['norm'],
#     device=config['device']
# ).to(config['device'])
model = TransH(
    ent_num=len(ent_map),
    rel_num=len(rel_map),
    dim=config['dim'],
    margin=config['margin'],
    norm=config['norm'],
    device=config['device']
).to(config['device'])
# 数据加载器
train_dataset = KGDataset(train_triples)
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

# 训练循环
best_hit10 = 0
best_mean_rank = float('inf')
for epoch in range(config['n_epoch']):
    model.train()
    total_loss = 0

    # 使用tqdm显示epoch进度条
    epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['n_epoch']}")
    for batch_idx, batch in enumerate(epoch_progress):
        batch = batch.to(config['device'])
        h, r, t = batch[:,0], batch[:,1], batch[:,2]

        # 生成负样本
        neg_t = torch.randint(0, len(ent_map), (len(batch),)).to(config['device'])
        neg_h = torch.randint(0, len(ent_map), (len(batch),)).to(config['device'])

        # 计算距离
        pos_dist = model(h, r, t)
        neg_dist_t = model(h, r, neg_t)
        neg_dist_h = model(neg_h, r, t)

        # 计算损失
        loss = model.loss(pos_dist, neg_dist_t) + model.loss(pos_dist, neg_dist_h)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新batch进度条的描述
        epoch_progress.set_postfix(loss=loss.item())

    # 验证评估
    valid_tensor = collate_fn(valid_triples).to(config['device'])
    hit10, mean_rank = model.evaluate(valid_tensor)
    
    print(f"Epoch {epoch+1}: Loss={total_loss:.2f}, Valid Hit@10={hit10:.4f}, Mean Rank={mean_rank:.1f}")
    
    # 保存最佳模型
    if mean_rank < best_mean_rank:
        best_mean_rank = mean_rank
        torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
torch.save(model.state_dict(), f"{output_dir}/final_model.pth")
# 最终测试
test_triples = load_triples(f"{config['data_dir']}/test.txt", ent_map, rel_map)
test_tensor = collate_fn(test_triples).to(config['device'])
hit10, mean_rank = model.evaluate(test_tensor)
print(f"\nFinal Test Results: Hit@10={hit10:.4f}, Mean Rank={mean_rank:.1f}")

model.load_state_dict(torch.load(output_dir, weights_only=True))
hit10, mean_rank = model.evaluate(test_tensor)
print(f"\nFinal Best Results: Hit@10={hit10:.4f}, Mean Rank={mean_rank:.1f}")

n2n_triples = load_id_triples(f"{config['data_dir']}/n-n.txt")
n2n_tensor = collate_fn(n2n_triples).to(config['device'])
head, tail = model.evaluate(n2n_tensor, split=True)
print(f"\nN-N Test Results: Head Hit@10={head:.4f}, Head Hit@10={tail:.4f}")
