import csv
import string
import torch
import numpy as np
import pandas as pd

time_records, scores = [], []
time_records_test, scores_test = [], []
rate = []
time_records_list = 0
event = []


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


def read3(path):
    with open(path) as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            # record
            # if 1 < i < 3:
            temp = row[2].split(" ")
            temp.pop()
            record = [int(x) for x in temp]
            # time_records.append(record)  # 双重列表
            # diff_1 = np.diff(record)
            # diff_2 = np.diff(diff_1)
            # print(i)
            # print(diff_1)
            # print(diff_2)
            # rate
            # rate.append(row[2])

            # score
            if int(row[1]) <= 33:  # split into three kinds
                row[1] = '1'
            elif int(row[1]) <= 66:
                row[1] = '2'
            else:
                row[1] = "3"
            # scores.append(row[1])
            # row[2] = row[2] + " " + row[3] + " " + row[4] + " " + row[5] + " " + row[6] + " " + row[7] + " " + row[8]
            event_code_first(record, row[3].strip("'"), i)
            i = i + 1
    # evnet去噪
    # 0.把第一个pl事件之前的事件全都移除？
    # 1.若果某个事件的下一个事件 距离该事件的时间小于三秒，那么就删除该事件。
    # 2.如果该事件如下一个事件是同类型，且间隔时间小于五秒，则合并为一个事件。
    # print(event[6])
    for index_1, u_v_pair in enumerate(event):  # 取出一条轨迹的时间序列，用u_v_pair列表的存储
        pop_list_1 = []  # 把需要删除的index存储到一个list中，在for-in循环结束后再统一删除。
        pop_list_2 = []  # 在for-in中实时删除，会导致u_v_pair和event都实时变化，而index继续增加，到时有index没访问到

        # 0.
        for index_2, element in enumerate(u_v_pair):  # 取出每个u_v_pair中的一个事件
            if element[0] != 0:  # 如果不为0，则删除
                # u_v_pair.pop(index_2)
                # event[index_1].pop(index_2)
                pop_list_1.append(index_2)
            else:
                break
        index_helper = 0
        for ele_1 in pop_list_1:
            event[index_1].pop(ele_1-index_helper)
            index_helper = index_helper + 1

        # 1，2
        for index_2, element in enumerate(u_v_pair):  # 取出每个u_v_pair中的一个事件
            if index_2 > 0:
                element_before = event[index_1][index_2 - 1]
                timespan = element[2] - element_before[2]  # 计算该事件与前一个事件的相差时间
                if element[0] == element_before[0]:  # 如果是同类型
                    if timespan < 5:  # 比较发生的时间
                        element[2] = element_before[2]  # 将第二个事件的时间修改为第一个事件的时间
                        # event[index_1].pop(index_2 - 1)  # 删除前一个事件
                        pop_list_2.append(index_2 - 1)
                # 同类型小于三秒则删除前一个，会导致播放暂停出现问题？？？
                # else:  # 如果不是同类型
                #     if timespan < 3:
                #         # event[index_1].pop(index_2 - 1)  # 删除前一个事件
                #         pop_list_2.append(index_2 - 1)
        index_helper = 0
        for ele_2 in pop_list_2:
            event[index_1].pop(ele_2 - index_helper)
            index_helper = index_helper + 1
    # print(event[6])


def read_test():
    for i in range(0, 10):
        time_records_test.append(time_records[i])
        scores_test.append(scores[i])


all_letters = " 1234567890"
n_letters = len(all_letters)


def event_code_first(record, rate_str, event_i):
    # event新增一个index为event_i的项来存储第i个轨迹，并且把轨迹的第一个事件初始化为全都是-1的。
    # 因为不能保证第一个事件一定是pl事件，如果是其他事件而又没有初始化，则会导师在设置该事件时，其status的赋值发生不可访问性错误
    # event.append([])
    event.append([[-1, -1, -1, -1, -1]])
    if rate_str.strip(" ") == '':  # rate 为空：说明没有倍速，只有播放、暂停、快进、快退
        # event_code_no_rate(record, event_i)
        event_code_with_rate(record, 0, event_i)
    else:  # rate 不为空：
        # print(rate_str)
        # print(record)
        # print(np.diff(record))
        rate_list = rate_str.strip(" ").split(';')  # 用于保存rate的list
        # rate_list.pop()  # 去除最后一个空字符,或者将循环次数减少1
        # print(rate_list)
        rate_len = len(rate_list)
        record_left_edge = 0  # 用于标记分段处理的record的左边界
        i = 0  # 控制读取rate列表
        rate_ele_before = 0  # 记录上一次的rate，可以先初始化为0
        while i < rate_len-1:
            pair = rate_list[i].split(' ')
            rate_ele = float(pair[0])  # 保存所改变的速率
            sec = int(pair[1])  # 保存改变速率发生的位置
            # 下面这个if-else，是处理sec有可能不在record中的情况
            if sec in record:
                sec = sec
            else:
                for t in range(1, 5):
                    if sec-t in record:
                        sec = sec-t
                        break
                    elif sec+t in record:
                        sec = sec+t
                        break
            print(event_i)
            print(record)
            record_range = record[record_left_edge: record.index(sec)+1]  # 获取record_range：截取当前sec与上一次边界之间的record，来进行处理

            # 1.给定length值：根据不同情况赋值不同的length
            if record_left_edge == 0:  # 只变了一次速度的最后一次，需要处理两段。也可以用record_left_edge == 0来判断
                if len(record_range) > 1:  # 长度大于1（一般情况下其都会），则正常赋值length = 0
                    length = 0
                else:  # 特出情况：首个range长度很短。代表用户在开始首先就改变了速度。则主动为其添加一个pl事件（本else语句中），变速事件在后面的循环语句中会被添加
                    print("刚开始就变速？")
                    length = -1  # 随便赋值，这段record_range不需要处理
                    # 添加Pl事件（尽量在变速事件之前，修改position和time参数都减少1）
                    if record.index(sec) > 0:
                        event[event_i].append([0, record[record.index(sec) - 1], record.index(sec), 1, 1])
                    else:  # 如果变速时的sec为0，那么pl事件和变速事件的position只能是一样的了
                        # print(event_i)
                        # print(len(event))
                        # print(event[event_i])
                        event[event_i].append([0, record[record.index(sec)], record.index(sec), 1, 1])
            else:  # 改变了多次速度的最后一次
                # 处理sec前的record_range,正常赋值length即可
                length = record.index(sec) + 1 - len(record_range)  # record_range的左边界

            # 该if-else语句是判断应该将range进行常规处理还是非常规（变速）处理.由于与非常规处理的差异不明显，所以最后其实都是一个函数。
            if rate_ele_before == 1.0 or record_left_edge == 0:  # 上一次操作重新切回正常速率。
                event_code_with_rate(record_range, length, event_i)  # 常规处理
            else:  # 开始变化成非正常速率。
                event_code_with_rate(record_range, length, event_i)  # 非常规处理

            # 该if-else语句是判断应该在处理range之后，将该次变速动作添加为什么事件
            # question：为什么这里添加事件时，用的都是record来定位，而在with_rate中采用的是裁切的record_range, 好像并无差别？？？
            if rate_ele > 1.0:
                event[event_i].append(
                    [4, record[record.index(sec)], record.index(sec) + 1, event[event_i][-1][3],
                     rate_ele])  # 添加Rf事件
            elif rate_ele < 1.0:
                event[event_i].append(
                    [5, record[record.index(sec)], record.index(sec) + 1, event[event_i][-1][3],
                     rate_ele])  # 添加Rs事件
            else:
                event[event_i].append(
                    [6, record[record.index(sec)], record.index(sec) + 1, event[event_i][-1][3],
                     rate_ele])  # 添加Rd事件

            # 最后一段需要单独处理
            if i == (rate_len - 1):  # 最后一次改变速率
                # 2.处理sec后的最后一段record_range_last：需要构造record_range_last和length_last
                record_range_last = record[record.index(sec) + 1:]
                length_last = record.index(sec) + 1  # # record_range的右边界
                rate_ele_last = rate_ele
                # 该if-else语句是判断应该将range进行常规处理还是非常规（变速）处理.由于与非常规处理的差异不明显，所以最后其实都是一个函数。
                if rate_ele_last == 1.0:
                    event_code_with_rate(record_range_last, length_last, event_i)  # 常规处理
                else:
                    event_code_with_rate(record_range_last, length_last, event_i)  # 非常规处理

            record_left_edge = record.index(sec) + 1  # 记录下一次range的左边缘index
            rate_ele_before = rate_ele  # 记录该速率，传给下一次循环。当该速率为正常速率，则下一次截取的range，应该正常处理
            i = i + 1


# 该函数处理非常规速度的record_range,并且接受record_range在该record之前的长度信息length_before，以及轨迹序号event_i
def event_code_with_rate(record_range, length_before, event_i):
    # if len(record_range) > 1:  # 当record_range太小时，就不做以下处理了。默认为这一点record中没有重要事件信息
    if length_before != -1:  # length_before = -1,说明遇到了刚开始就变速的情况，不需要处理
        record_diff_1 = np.diff(np.array(record_range))
        record_len = len(record_diff_1)
        # [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        #  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        #  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2
        #  1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1]
        record_diff_2 = np.diff(record_diff_1).tolist()
        for index, ele in enumerate(record_diff_2):  # 处理diff_2为一条条事件
            # if len(event[event_i]) > 0 and event[event_i][0][0] != 0:
            #     print("轨迹第一个事件不为pl事件，错误！！")
            if ele == 1:
                # if record_diff_1[index] == 1:  # i+2,
                if record_diff_1[index] == 0:
                    if (index > 0 and record_diff_1[index - 1] == 0) or index == 0:  # pl事件.需要分情况考虑index是否为0
                        event[event_i].append([0, record_range[index + 1], length_before + index + 1, 1, 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                elif record_diff_1[index] > 2:  # 两个Sf快进事件, 第二个下一步可以检测，这里就不处理
                    event[event_i].append([3, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                elif record_diff_1[index] == -1:  # sb + pa，下一步的pa检测不出，所以这里要处理
                    event[event_i].append([2, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                    event[event_i].append([1, record_range[index + 1], length_before + index + 1, 0, 1])
                elif record_diff_1[index] < -1:
                    event[event_i].append([2, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
            elif ele == -1:
                # if record_diff_1[i] == 0:  # Sb事件,i+2
                if record_diff_1[index] == 1:  # pa事件
                    if index + 2 < record_len:
                        if record_diff_1[index + 2] == 0:
                            event[event_i].append([1, record_range[index + 1], length_before + index + 1, 0, 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                elif record_diff_1[index] > 2:  # Sf事件  稍有变动
                    event[event_i].append([3, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                elif record_diff_1[index] < 0:  # Sb事件
                    event[event_i].append([2, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
            elif ele > 1:
                # if record_diff_1[index] == 0:  # Sf事件，i+2步的，不跨步执法
                # if record_diff_1[index] == 1:  # Sf事件，i+2步的，不跨步执法
                if record_diff_1[index] < 0:  # Sb事件
                    # print(len(record_range))
                    # print(index+1)
                    # print(event[event_i])
                    event[event_i].append([2, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                    if record_diff_1[index + 1] == 0:  # pa事件,i+2步的pa事件检测不出来，所以要在这一步处理
                        if index + 2 < record_len:
                            if record_diff_1[index + 2] == 0:
                                event[event_i].append([1, record_range[index + 1], length_before + index + 1, 0, event[event_i][-1][4]])
                elif record_diff_1[index] > 2:  # Sf事件, 可以变动也可以不变动。因为倍速的情况不会突然从1变成4，跨度不会达到3，diff_2不会达到2
                    event[event_i].append([3, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                elif record_diff_1[index-1] == 0 and record_diff_1[index] == 0 and record_diff_1[index+1] == 2 and (record_diff_1[index+2] == 1 or record_diff_1[index+2] == 2):  # pl事件
                    event[event_i].append([0, record_range[index + 1], length_before + index + 1, 1, 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
            elif ele < -1:
                # if record_diff_1[index] == 0:  # Sf事件，i+2步的，不跨步执法
                # if record_diff_1[index] == 1:  # Sf事件，i+2步的，不跨步执法
                if record_diff_1[index] < 0:  # Sb事件,record_sorted.csv
                    event[event_i].append([2, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                elif record_diff_1[index] > 2:  # Sf事件, 稍有变动
                    if len(event[event_i]) == 0:
                        print("轨迹第一个事件不为pl事件，错误！！")
                        print(record_diff_2)
                        print(record_diff_1)
                    else:
                        event[event_i].append([3, record_range[index + 1], length_before + index + 1, event[event_i][-1][3], 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])
                    if record_diff_1[index + 1] == 0:  # pa事件,i+2步的pa事件检测不出来，所以要在这一步处理
                        if index + 2 < record_len:
                            if record_diff_1[index + 2] == 0:
                                event[event_i].append([1, record_range[index + 1], length_before + index + 1, 0, 1 if event[event_i][-1][4]==-1 else event[event_i][-1][4]])


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


if __name__ == '__main__':
    read3("data/sorted_time_record_database.csv")

    name = ['event', 'position', 'timestamp', 'status', 'rate']
    event_frame_list = [pd.DataFrame(columns=name, data=event_i) for event_i in event]
    keys = [str(event_i) for event_i in range(0, len(event_frame_list))]
    event_frame = pd.concat(event_frame_list, keys=keys)
    event_frame.to_csv("data/event_coded_time_record_db.csv", encoding='gbk')
