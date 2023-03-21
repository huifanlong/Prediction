import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import torch
import torch.nn as nn


#############################################################
# 读取文件，进行词典的创建，并获得输入的event和score


class Dictionary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


eventDictionary = Dictionary("event")


def prepareData():
    # 处理输入的event
    event_frame = pd.read_csv("data/data_delete_short_30_event.csv")  # 读取event，其格式是dataframe
    event_frame_list = [event_frame[event_frame['Unnamed: 0'] == event_i].iloc[:, 2:] for event_i in
                        range(0, event_frame.iloc[len(event_frame) - 1, 0]+1)]  # 将event转为list
    event_series_list = []
    for index, lis in enumerate(event_frame_list):  # 每个获得的lis是一个dataframe，其中包含了一条点击流数据的所有event
        series = lis["event"].map(str).str.cat(
            [lis["position"].map(str), lis["timestamp"].map(str), lis["status"].map(str), lis["rate"].map(str)],
            sep=";", na_rep="")  # 将五个列属性合并为一个string类型
        event_series_list.append(series.reset_index(drop=True))  # 每个series其实都带有原来的index(已修改)
        for ele in series:  # 每个string类型的ele类似'0;0;3;1;1.0'，将其当做一个单词来进行比较
            eventDictionary.addWord(ele)
    # 处理输入的score
    str_frame = pd.read_csv("data/data_delete_short_30_str.csv", names=["str", "score", "rate"])
    score = str_frame["score"]
    return event_series_list, score


input_event, input_score = prepareData()  # 这里的input_event是一个series类型的list,input_score是一个series
# 打印出现次数较多的单词
# for key, value in eventDictionary.word2count.items():
#     if value > 1:
#         print("% %", key, value)


#############################################################
# 进行训练

# 构建数据集
class CustomEmbeddingDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.input_indexes_list = []
        self.target_index_list = []
        for event_ele in input_event:
            if len(event_ele) > 1:
                for i in range(0, len(event_ele)):
                    a = event_ele[i]
                    target_index = tensorFromEventSingle(event_ele[i])
                    input_indexes = tensorFromEventList(event_ele.drop([i]))
                    self.target_index_list.append(target_index)
                    self.input_indexes_list.append(input_indexes)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.input_indexes_list)

    def __getitem__(self, idx):
        input_ides = self.input_indexes_list[idx]
        target_id = self.target_index_list[idx]
        if self.transform:
            input_ides = self.transform(input_ides)
        if self.target_transform:
            target_id = self.target_transform(target_id)
        return input_ides, target_id


def my_collate(batch):
    score = []
    event = []
    for sample in batch:
        event.append(sample[0])  # 其是一个series
        score.append(sample[1])  # 其是一个int
    return event, score


class EmbeddingRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(EmbeddingRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.embed_size)
        self.relu = nn.ReLU()

    def forward(self, input_index, target_index, hidden):
        embedded_input = self.embedding(input_index).squeeze(1)
        embedded_target = self.embedding(target_index).squeeze(1)
        output, hidden = self.gru(embedded_input, hidden)
        predicted = self.linear(hidden)
        predicted_relu = self.relu(predicted)
        return predicted_relu, embedded_target

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def indexesFromEventList(event_list):
    return [eventDictionary.word2index[event] for event in event_list]


def tensorFromEventSingle(event):
    index = eventDictionary.word2index[event]
    return torch.tensor(index, dtype=torch.long).view(-1, 1)


def tensorFromEventList(event_list):
    indexes = indexesFromEventList(event_list)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def train(module, criterion, optimizer, input_indexes_batch, target_index_batch):
    hidden = module.initHidden()

    optimizer.zero_grad()

    loss = 0

    for i in range(0, len(input_indexes_batch)):
        predicted_tensor, target_tensor = module(input_indexes_batch[i], target_index_batch[i], hidden)
        loss += criterion(predicted_tensor, target_tensor)

    loss.backward()

    optimizer.step()

    return loss


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def trainIter(n_iters, print_every=100, plot_every=10):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    embedding_module = EmbeddingRNN(eventDictionary.n_words, embed_size=5, hidden_size=128)
    criterion = nn.MSELoss()
    # 模型里面有embedding层，该层的event的向量表示应该也是传入了optimizer中，代表需要进行更新的参数
    module_optimizer = torch.optim.SGD(embedding_module.parameters(), lr=0.01)
    # embedding_optimizer = torch.optim.SGD(embedding_module.parameters(), lr=0.01)

    training_data = CustomEmbeddingDataset()
    dataloader = DataLoader(training_data, batch_size=10, collate_fn=my_collate, shuffle=True,)
    for iter in range(1, n_iters+1):
        for batch, (input_indexes_lis, target_index_lis) in enumerate(dataloader):
            if batch == n_iters:
                break
            else:
                loss = train(embedding_module, criterion, module_optimizer, input_indexes_lis, target_index_lis)

                print_loss_total += loss
                plot_loss_total += loss
                # print("ok")

                if (batch+1) % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, (batch+1) / n_iters),
                                                 (batch+1), (batch+1) / n_iters * 100, print_loss_avg))

                if (batch+1) % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
            # print("ok")

    showPlot(plot_losses)
    torch.save(embedding_module.state_dict(), 'model/model_weights.pth')


trainIter(100)


# training_data = CustomTraceDataset()
# weights = [7 if score == 0 else 4 if score == 1 else 1 for record, score in training_data]
# sampler = WeightedRandomSampler(weights, num_samples=10000, replacement=True)
# dataloader = DataLoader(training_data, batch_size=1, sampler=sampler, collate_fn=my_collate)
# for batch, (event_list, score_list) in enumerate(dataloader):
#     print("1")

# print("ok")
