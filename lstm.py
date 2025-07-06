import torch
import torch.nn as nn
from torchtext.data import Field, Example, Dataset, Iterator
import jieba
import numpy as np

# 分词函数
def tokenize(text):
    return list(jieba.cut(text))

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeddings):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, input_lengths):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        output = self.fc(hidden.squeeze(0))
        return output

def get_lstm_features(text):
    # 创建Field对象，用于定义文本的预处理和特征
    text_field = Field(tokenize=tokenize, lower=True, include_lengths=True, batch_first=True)
    text_field.build_vocab([text])
    # 创建Example对象
    example = Example.fromlist([[text]], [('text', text_field)])
    # 创建Dataset对象
    dataset = Dataset([example], [('text', text_field)])
    # 创建Iterator对象
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iterator = Iterator(dataset, batch_size=batch_size, device=device)
    # 加载预训练的中文词向量
    embedding_size = 512  # 假设预训练词向量维度为300
    pretrained_embeddings = np.random.randn(len(text_field.vocab), embedding_size)  # 替换为您的预训练词向量
    # 定义模型超参数
    input_size = embedding_size
    hidden_size = 768
    output_size = 512
    # 初始化模型
    model = LSTMModel(input_size, hidden_size, output_size, pretrained_embeddings).to(device)
    # 提取文本特征
    input_seq, input_lengths = next(iter(iterator)).text
    feature_vector = model(input_seq, input_lengths).detach().cpu().numpy()[0]
    return feature_vector
