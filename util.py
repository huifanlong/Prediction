import torch

from dataProcessing import *
from torch.utils.data import Dataset
import time
import math
import random
import pandas as pd

min_str_length = 30


# dataset1：原始的str轨迹数据作为输入
class CustomTraceDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.records = pd.read_csv(csv_file)
        # 将轨迹str长度低于某指定最低长度min_str_length的数据清除
        self.records = self.records[[len(self.records['str'][index].strip().split(" ")) > min_str_length for index in range(0, len(self.records))]]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        time_record_init = self.records.values[idx, 0]
        time_record_init = list_number_to_tensor(time_record_init.strip().split(" "))  # 新的构建方式
        # time_record_init = line_to_tensor(time_record_init) # 初始构建方式
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


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() + 1
    return category_i  # 其值为1，2，3


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def number_to_tensor(num):
    result = ""
    while num != 0:
        ret = int(num % 2)
        num = int(num / 2)
        result = str(ret) + result
    # print(len(result))
    result = ("0" * (11 - len(result)) + result) if len(result) < 11 else result
    tensor = torch.zeros(1, 11)
    for index, ele in enumerate(list(result)):
        tensor[0][index] = int(ele)
    return tensor


def list_number_to_tensor(list_num):
    tensor = torch.zeros(len(list_num), 1, 11)
    for index, ele in enumerate(list_num):
        tensor[index] = number_to_tensor(int(ele))
    return tensor

# def random_choice(l):
#     return l[random.randint(0, len(l) - 1)]


# def random_training_example():
#     index = random.randint(0, len(time_records) - 1)
#     # print(index)
#     category = scores[index]
#     line = time_records[index]
#     # print(category)
#     # print(line)
#     category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)  # tensor([2])?
#     line_tensor = line_to_tensor(line)
#     # print(category_tensor)
#     # print(line_tensor)
#     return category, line, category_tensor, line_tensor


# all_categories = ["1", "2", "3"]
