import sys

import numpy as np

from dataProcessing import *
from model import *
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

from util import CustomTraceDataset, category_from_output, my_collate

n_hidden = 128
n_categories = 3
n_epochs = 200
print_every = 50
plot_every = 10
learning_rate = 0.0035  # If you set this too high, it might explode. If too low, it might not learn
# all_categories = ["1", "2", "3"]


rnn = RNN(n_letters, n_hidden, n_categories)
# optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.5, dampening=0.5)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
# optimizer = torch.optim.adam
# swa：
swa_model = AveragedModel(rnn)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
swa_start = 7
swa_scheduler = SWALR(optimizer, swa_lr=0.05)


def train(dataloader):
    rnn.train()
    total_acc, total_count, total_num, total_loss = 0, 0, 0, 0
    log_interval = 25
    start_time = time.time()
    # hidden = rnn.init_hidden()
    # optimizer.zero_grad()

    for batch, (records_batch, scores_batch) in enumerate(dataloader):
        idx_batch = 0
        for record in records_batch:
            hidden = rnn.init_hidden()
            optimizer.zero_grad()
            for i in range(record.size()[0]):
                output, hidden = rnn(record[i], hidden)  # tensor：(1,3)
                assert not torch.any(torch.isnan(output))  # assert，对训练过程中的数据进行检查，可以精确定位，一般是对输出结果和loss进行判断
            target = scores_batch[idx_batch]  # tensor：(1,) 具体是0,1,2
            loss = criterion(output, target)  # 这里表示是在每一条轨迹最后的输出才计算loss，如果叠加其loss是否会让模型更准确呢？
            if torch.isnan(loss):
                sys.exit()
            int_target = target.item()+1
            """
            # Example of target with class probabilities
            # input = torch.randn(3, 5, requires_grad=True)
            # target = torch.randn(3, 5).softmax(dim=1)
            # loss = F.cross_entropy(input, target)
            # loss.backward()
            """
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1)  # 梯度裁剪
            # if batch % log_interval == 24 and batch > 0:
            #     for name_before, parms_before in rnn.named_parameters():
            #         print('=====迭代zhihou=====梯度:', parms_before.grad)
            optimizer.step()
            # 以上即是train一条学生轨迹的所有步骤
            # 以下是做一些个训练过程记录
            total_acc = total_acc + 1 if category_from_output(output) == int_target else total_acc
            # print('output_tensor {},output {},target {} ,total_acc {}'.format(output, category_from_output(output),
            # int_target, total_acc))

            total_count = total_count+1
            total_loss += loss.item()
            idx_batch = idx_batch + 1
        if batch % log_interval == 24 and batch > 0:
            elapsed = time.time() - start_time
            # for name_after, parms_after in rnn.named_parameters():
            #     print('=====更新之后=====梯度:', parms_after.grad)
            print('| loss {:3} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(total_loss, batch+1, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count, total_loss = 0, 0, 0
            start_time = time.time()


# 采用权重采样 在907个样本中按权重重负采样8000次（10个iter）
# 通过自定义的dataset类来构建dataset，这里只需要传入csv文件路径，可以返回record（已转成tensor）和score（未转成tensor的int类型）
training_data = CustomTraceDataset('/home/hadoop/PycharmProjects/Prediction_vertion1/data/prediction_version1.csv')
# 该方法是设置一个权重向量，为将要选取的每个样本赋予不同的权重。该方法会遍历执行len(training_data)次__getitem__函数，且是按顺序的idx
weights = [7 if score == 0 else 4 if score == 1 else 1 for record, score in training_data]
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
    if epoch > swa_start:
        swa_model.update_parameters(rnn)
        swa_scheduler.step()
    else:
        scheduler.step()

# optimizer.swap_swa_sgd()
torch.optim.swa_utils.update_bn(dataloader, swa_model)
torch.save(rnn, 'model/char-rnn-classification_saw.pt')
print("Done")
