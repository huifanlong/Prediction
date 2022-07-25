from dataProcessing import *
from torch.utils.data import Dataset
import time
import math
import random
import pandas as pd


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
