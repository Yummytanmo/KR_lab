import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import wandb
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=31, vocab_size=30000):
        with open(file_path, 'r', encoding='gb2312', errors='ignore') as f:
            text = f.read()
        tokens = list(jieba.lcut(text))
        
        counter = Counter(tokens)
        most_common = counter.most_common(vocab_size - 2)
        vocab = ['<pad>', '<unk>'] + [word for word, _ in most_common]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}

        self.data = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_length]
        target = self.data[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.lstm(embeds, hidden)
        logits = self.fc(output)
        return logits, hidden

def train(seq_length=31, vocab_size=30000, embed_size=64, hidden_size=128, batch_size=64, num_epochs=10, learning_rate=0.001):
    run_name = f"seq{seq_length}_vocab{vocab_size}_embed{embed_size}_hidden{hidden_size}_batch{batch_size}_epochs{num_epochs}_lr{learning_rate}"
    wandb.init(project="KR-lab1-LSTM", name=run_name, config={
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
    })

    dataset = TextDataset('./data/text.txt', seq_length, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTM(vocab_size, embed_size, hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wandb.watch(model, log="all")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    example_ct = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch_idx, (batch_x, batch_y) in enumerate(progress_bar):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            example_ct += batch_x.size(0)
            wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1}, step=example_ct)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
        
    torch.save(model.state_dict(), 'lstm_model_final.pth')
    wandb.finish()

if __name__ == "__main__":
    train(num_epochs=6)
