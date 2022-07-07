import torch
from data import *
from dataProcessing import *
from model import *
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
import random
import pandas as pd
import time
import math

n_hidden = 128
n_categories = 3
n_epochs = 10
print_every = 50
plot_every = 10
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
all_categories = ["1", "2", "3"]


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() + 1
    return category_i  # 其值为1，2，3


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example():
    index = random.randint(0, len(time_records) - 1)
    # print(index)
    category = scores[index]
    line = time_records[index]
    # print(category)
    # print(line)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)  # tensor([2])?
    line_tensor = line_to_tensor(line)
    # print(category_tensor)
    # print(line_tensor)
    return category, line, category_tensor, line_tensor


rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def my_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    text = []
    for sample in batch:
        text.append(sample[0])
        # targets.append(line_to_tensor(str(sample[1])))
        targets.append(torch.tensor([sample[1]], dtype=torch.long))
    return text, targets


class CustomTraceDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.records = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        time_record_init = self.records.values[idx, 0]
        time_record_init = line_to_tensor(time_record_init)
        score_init = self.records.values[idx, 1]
        if score_init <= 33:
            score_init = 0
        elif score_init <= 66:
            score_init = 1
        else:
            score_init = 2
        if self.transform:
            time_record_init = self.transform(time_record_init)
        if self.target_transform:
            score_init = self.target_transform(score_init)
        return time_record_init, score_init  # score_init:tensor([0])、tensor([1])、tensor([2])


def train(dataloader):
    rnn.train()
    total_acc, total_count, total_num, total_loss = 0, 0, 0, 0
    log_interval = 25
    start_time = time.time()
    hidden = rnn.init_hidden()
    optimizer.zero_grad()

    for batch, (records_batch, scores_batch) in enumerate(dataloader):
        idx_batch = 0
        for record in records_batch:
            hidden = rnn.init_hidden()
            optimizer.zero_grad()

            for i in range(record.size()[0]):
                output, hidden = rnn(record[i], hidden)  # tensor：(1,3)
            target = scores_batch[idx_batch] # tensor：(1,)
            loss = criterion(output, target)  # 这里表示是在每一条轨迹最后的输出才计算loss，如果叠加其loss是否会让模型更准确呢？
            """
            # Example of target with class probabilities
            # input = torch.randn(3, 5, requires_grad=True)
            # target = torch.randn(3, 5).softmax(dim=1)
            # loss = F.cross_entropy(input, target)
            # loss.backward()
            """
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.1)  # 这步是新加的要不要保留 还不确定
            loss.backward()
            optimizer.step()
            # 以上即是train一条学生轨迹的所有步骤
            # 以下是做一些个训练过程记录
            total_acc = total_acc + 1 if category_from_output(output) == scores_batch[idx_batch].item() else total_acc
            total_count += len(records_batch)
            total_loss += loss.item()
            idx_batch = idx_batch + 1
        if batch+1 % log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| loss {:3} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(total_loss, batch+1, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count, total_loss = 0, 0, 0
            start_time = time.time()


# 采用权重采样 在907个样本中按权重重负采样8000次（10个iter）
# 通过自定义的dataset类来构建dataset，这里只需要传入csv文件路径，可以返回record（已转成tensor）和score（未转成tensor的int类型）
training_data = CustomTraceDataset('/home/hadoop/PycharmProjects/Prediction_vertion1/data/prediction_version1.csv')
# 该方法是设置一个权重向量，为将要选取的每个样本赋予不同的权重。该方法会遍历执行len(training_data)次__getitem__函数，且是按顺序的idx
weights = [9 if score == 1 else 5 if score == 2 else 1 for record, score in training_data]
# num_sampler控制其每一次构造dataloader能够有50个batch
sampler = WeightedRandomSampler(weights, num_samples=16 * 50, replacement=True)

for epoch in range(1, n_epochs + 1):
    epoch_start_time = time.time()
    dataloader = DataLoader(training_data, batch_size=16, sampler=sampler, collate_fn=my_collate)
    # dataloader加载的数量是由参数sampler中的所sampler样本数量决定
    # 改写了dataloader中的collate方法，返回的batch并没有将其stack，因为该模型用的是RNN，输入文本长度不一致。\
    # 而dataloader的常规做法是将一个batch的feature（通常是相同大小的图片）stack成一个长的tensor统一处理。\
    # 但是此处利用dataloader的sampler实现了样本类型失衡的按不同权重取样本的策略
    train(dataloader)
    # category, line, category_tensor, line_tensor = random_training_example()
    # output, loss = train(category_tensor, line_tensor)
    # if guess == int(category):
    #     correct_num = correct_num+1
    # if guess == 3:
    #     guess_3 = guess_3 + 1
    # elif guess == 2:
    #     guess_2 = guess_2 + 1
    # total_num = total_num + 1
    # Print epoch number, loss, name and guess

torch.save(rnn, 'char-rnn-classification_new.pt')
print("Done")
