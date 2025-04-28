# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            name, eid = line.strip().split('\t')
            mapping[name] = int(eid)
    return mapping

def load_triples(file_path, ent_map, rel_map):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, t, r = line.strip().split('\t')[:3]
            try:
                triples.append((
                    ent_map[h.strip()],
                    rel_map[r.strip()],
                    ent_map[t.strip()]
                ))
            except KeyError as e:
                print(f"Ignored invalid triple: {line.strip()}")
    return triples

def load_id_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 3:
                triples.append((int(line[0]), int(line[2]), int(line[1])))  # Convert to integers
    return triples

class KGDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]

def collate_fn(batch):
    return torch.tensor(batch, dtype=torch.long)