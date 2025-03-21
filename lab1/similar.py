import argparse
from gensim.models import word2vec
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontManager
import matplotlib

# Parse command line arguments
parser = argparse.ArgumentParser(description='Word2Vec training and similarity calculation')
parser.add_argument('--window', type=int, default=3, help='Window size')
parser.add_argument('--vector_size', type=int, default=32, help='Embedding vector size')
parser.add_argument('--sg', type=int, default=1, help='Use skip-gram (1) or CBOW (0)')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Load corpus
sentences = word2vec.Text8Corpus('./data/seg_text.txt')

# Train model
model = word2vec.Word2Vec(sentences, window=args.window, vector_size=args.vector_size, epochs=args.epochs, seed=args.seed, sg=args.sg)
model.save("./ckpt/1019.model")

matplotlib.rc("font",family='YouYuan')
model = word2vec.Word2Vec.load('./ckpt/1019.model')

# Calculate and print similarities
sim_word1_word2 = model.wv.similarity("岳不群", "林平之")
sim_word1_word3 = model.wv.similarity("岳不群", "岳灵珊")

print("Similarity between '岳不群' and '林平之':", sim_word1_word2)
print("Similarity between '岳不群' and '岳灵珊':", sim_word1_word3)
