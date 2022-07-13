from torch import nn
import torch

from data.word_dict import WordDict
from torch.utils.data import DataLoader
from data.Dataset import CoNLLDataset
from model import BI_LSTM_CRF_MODEL
from torch.optim import SGD

wordDict = WordDict('./data/glove.6B.100d.txt')
word2ID = wordDict.get_word2ID()
dataset = CoNLLDataset('./data/CoNLL-2003/train.txt', word2ID)
dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
pre_trained_embedding = wordDict.get_wordVector()
pre_trained_embedding = torch.Tensor(pre_trained_embedding)
tag_index = dataset.get_tag_index()
model = BI_LSTM_CRF_MODEL(vocab_size=len(word2ID), embedding_dim=pre_trained_embedding.size()[1], hidden_dim=64,
                          tag_index=tag_index, pre_trained_Embedding=pre_trained_embedding)

optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
MAX_EPOCH=1000

for epoch in range(MAX_EPOCH):
    for i, data in enumerate(dataloader):
        # print(data)
        sequence, tags = data
        sequence = sequence.squeeze(0)
        tags = tags.squeeze(0)
        model.zero_grad()
        loss = model(sequence, tags)
        print(loss)
        loss.backward()
        optimizer.step()
        # break

# for epoch in range(500):
#     for i, data in enumerate(dataloader):
#         print(data)
#         break
