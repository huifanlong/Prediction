import csv
import string
import torch

time_records, scores = [], []
time_records_test,scores_test = [], []


def read():
    with open("data/prediction_version1.csv") as f:
        reader = csv.reader(f)
        time_records2 = [row[0] for row in reader]
        scores2 = [row[1] for row in reader]  # bug comes here
        print(scores)
        del (time_records[0])

        # print(len(time_records))
        # print(time_records[0])


def read2():
    import pandas as pd

    data = pd.read_csv("data/prediction_version1.csv")  # 读取文件中所有数据
    # 按列分离数据
    time_records1 = data[['time']]
    print(time_records)
    scores1 = data[['score']]  # 读取某一列
    print(scores)
    #     score
    # 0       66
    # 1      100
    # 2      100
    # 3      100
    # 4      100
    # ..     ...
    # 903      0
    # 904    100
    # 905     50
    # 906     50
    # 907     50


def read3():
    with open("data/prediction_version1.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            time_records.append(row[0])
            if int(row[1]) <= 33:  # split into three kinds
                row[1] = '1'
            elif int(row[1]) <= 66:
                row[1] = '2'
            else:
                row[1] = "3"
            scores.append(row[1])
    # print(time_records)
    # print(scores)


def read_test():
    for i in range(0, 10):
        time_records_test.append(time_records[i])
        scores_test.append(scores[i])


all_letters = " 1234567890"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters + 1)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# " 1234567890" refers ro:
# tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
# tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
# tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
# tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
# tensor([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])


# Turn a line into a <line_length x 1 x n_letters>,
# eg:torch.Size([13, 1, 11])
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
