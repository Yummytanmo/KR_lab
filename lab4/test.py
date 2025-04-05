from model import BertNer
from data_loader import NerDataset
from config import NerConfig
from transformers import BertTokenizer
import json
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_test_data(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    data = [json.loads(d)["text"] for d in lines]
    return data

def process_sentence(sentence):
    config = NerConfig("duie")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", cache_dir=config.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertNer(config).to(device)
    state_dict = torch.load("./checkpoint/duie/pytorch_model_ner.bin", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    entities = []
    current_entity = None
    current_entity_type = None

    encoding = tokenizer(sentence, return_tensors="pt", max_length=config.max_seq_len, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predictions = output.logits[0]

    idx2label = config.id2label
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = [idx2label[idx] for idx in predictions]

    for token, label in zip(tokens, labels):
        if label != 'O':
            print(f"({token},{label})")
        if label.startswith('B'):
            if current_entity is not None:
                entities.append((current_entity_type, current_entity))
            current_entity = token
            current_entity_type = label[2:]
        elif label.startswith('I') and current_entity is not None:
            current_entity += token
        else:
            if current_entity is not None:
                entities.append((current_entity_type, current_entity))
                current_entity = None
                current_entity_type = None
    if current_entity is not None:
        entities.append((current_entity_type, current_entity))

    return {
        "sentence": sentence,
        "entities": entities
    }

def main():
    test_data = load_test_data("./duie_data/test_example.json")
    num_processes = 12
    print(num_processes)
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_sentence, test_data), total=len(test_data)))

    with open("./results/result_example.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
