# KR_lab

本项目为知识图谱与自然语言处理相关实验，包含多个实验任务，涵盖了词向量、命名实体识别、关系抽取、知识图谱构建等内容。每个实验均包含源码、数据集及运行脚本。

## 目录结构

```
KR_lab/
├── lab1/         # 实验1：词向量与文本分类
│   └── 包含 word2vec.py, BERT.py, train_cls.py, train_ner.py 等
├── lab2/         # 实验2：图像分类（MNIST）
│   └── 包含 model.py, MNIST.py, eval.py 等
├── lab3/         # 实验3：知识图谱构建
│   └── 包含 graphexp.ipynb, enzymes 数据集等
├── lab4/         # 实验4：命名实体识别
│   └── 包含 main.py, process.py, ner_llm.py 等
├── lab5/         # 实验5：关系抽取
│   └── 包含 OpenNER, CNN, BERT-Relation-Extraction 等
├── lab6/         # 实验6：知识图谱补全
│   └── 包含 train.py, Trans.py, Trainer.py 等
├── dataset/      # 统一数据集
├── glove/        # GloVe 预训练词向量
└── README.md     # 项目说明文档
```

## 实验说明

### Lab1：词向量与文本分类
- 使用 word2vec 和 BERT 进行词向量训练。
- 实现文本分类和命名实体识别任务。

### Lab2：图像分类（MNIST）
- 使用 PyTorch 实现 MNIST 图像分类模型。
- 包含模型训练、评估和可视化。

### Lab3：知识图谱构建
- 使用 Jupyter Notebook 进行知识图谱构建实验。
- 包含 enzymes 数据集的处理和可视化。

### Lab4：命名实体识别
- 实现命名实体识别模型，支持中文 NER。
- 包含数据预处理、模型训练和评估。

### Lab5：关系抽取
- 实现基于 CNN 和 BERT 的关系抽取模型。
- 支持 OpenNER 和自定义数据集。

### Lab6：知识图谱补全
- 使用 TransE 等模型进行知识图谱补全。
- 支持 WN18 和 FB15k 数据集。

## 构建与运行

以 Lab1 为例：

```bash
cd lab1
python train_cls.py   # 训练文本分类模型
python train_ner.py   # 训练命名实体识别模型
```

其他实验的构建和运行方式类似，具体可参考各实验目录下的 `run.sh` 或 `train.sh` 脚本。

## 依赖

- Python 3.6+
- PyTorch
- TensorFlow
- scikit-learn
- pandas
- numpy
- jupyter

## 贡献与许可

本项目仅用于课程学习与交流，禁止用于任何商业用途。