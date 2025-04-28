import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import load_mapping, load_triples, KGDataset, collate_fn
from TransE import TransE
from TransH import TransH
from TransR import TransR
config = {
    'data_dir': './FB15k',
    'n_epoch': 2000,
    'batch_size': 10240,
    'dim': 200,
    'margin': 1.0,
    'norm': 2,  # 1或2范数
    'lr': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
# 加载数据
ent_map = load_mapping(f"{config['data_dir']}/entity2id.txt")
rel_map = load_mapping(f"{config['data_dir']}/relation2id.txt")
# 创建模型
model = TransR(
    ent_num=len(ent_map),
    rel_num=len(rel_map),
    dim=config['dim'],
    margin=config['margin'],
    norm=config['norm'],
    device=config['device']
).to(config['device'])
# Load the best model checkpoint
model.load_state_dict(torch.load('./FB15kTransr_1000_200_1.0_2_0.001/best_model.pth', weights_only=True))

test_triples = load_triples(f"{config['data_dir']}/test.txt", ent_map, rel_map)
test_tensor = collate_fn(test_triples).to(config['device'])
hit10, mean_rank = model.evaluate(test_tensor)
print(f"\nFinal Test Results: Hit@10={hit10:.4f}, Mean Rank={mean_rank:.1f}")