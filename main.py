from dataProcessing import read3, event
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # min_str_length = 30
    # records = pd.read_csv("data/prediction_version1.csv")
    # # rate是由多列构成的，把多列的rate组合成一个rate
    # records["rate"] = records["rate"].str.cat(
    #     [records["Unnamed: 3"], records["Unnamed: 4"], records["Unnamed: 5"], records["Unnamed: 6"],
    #      records["Unnamed: 7"],
    #      records["Unnamed: 8"]], sep=";", na_rep="")
    # # 把合并之后不需要的列删除
    # records = records.drop(records.columns[3:], axis=1)
    # # 将轨迹str长度低于某指定最低长度min_str_length的数据清除
    # # records = records[[len(records['str'][index].strip().split(" ")) > min_str_length for index in range(0, len(records))]]
    # records["rate"] = records["rate"].map(lambda x: x.strip(";"))
    # # 进行event的encode
    # records.to_csv("data/data_all_str.csv", encoding='gbk', header=None, index=None)
    # read3("data/data_all_str.csv")
    #
    # name = ['event', 'position', 'timestamp', 'status', 'rate']
    # event_frame_list = [pd.DataFrame(columns=name, data=event_i) for event_i in event]
    # keys = [str(event_i) for event_i in range(0, len(event_frame_list))]
    # event_frame = pd.concat(event_frame_list, keys=keys)
    # event_frame.to_csv("data/data_all_event.csv", encoding='gbk')

    # 在训练过程中对存储event的文件进行解析
    # 统计event数量与分数的关系
    data = pd.read_csv("data/data_delete_short_30_event.csv")
    # pd = pd.DataFrame(columns=["str", "score", "rate"], data=pd.read_csv("data/data_all_str.csv"))
    p = pd.read_csv("data/data_delete_short_30_str.csv", names=["str", "score", "rate"])
    score = p["score"]
    str = p["str"]
    rate = p["rate"]
    # print(data.iloc[len(data)-1, 0])  # 获取第一列的最大数(即uvPair数目)
    # print(data[data['Unnamed: 0'] == 0].iloc[:, 2:])  # 获取uvPair为0的event序列
    num1, num2, num3to5, num6to10, numOver10 = 0, 0, 0, 0, 0
    score0, score1, score3 = 0, 0, 0
    score_list = []
    str_list = []
    rate_list = []
    event_frame_list = [data[data['Unnamed: 0'] == event_i].iloc[:, 2:] for event_i in range(0, data.iloc[len(data)-1, 0]+1)]
    for index, lis in enumerate(event_frame_list):
        if len(lis) == 1:
            num1 += 1
            score_list.append(score[index])
            rate_list.append(rate[index])
            str_list.append(str[index])
            if score[index] <= 33:
                score0 += 1
            elif score[index] <= 66:
                score1 += 1
            else:
                score3 += 1
        elif len(lis) == 2:
            num2 += 1
        elif 2 < len(lis) < 6:
            num3to5 += 1
        elif 5 < len(lis) < 11:
            num6to10 += 1
        else:
            numOver10 += 1
    pd.DataFrame(data=str_list, columns=["str"]).to_csv("data/event_equal_1_str_all.csv", index=False)
    pd.DataFrame(data=rate_list, columns=["rate"]).to_csv("data/event_equal_1_rate_all.csv", index=False)

    # 观察event事件只有一个的str
    # str = pd.read_csv("data/event_equal_1_str_30.csv")
    # rate = pd.read_csv("data/event_equal_1_rate_30.csv")
    # str_list = str["str"]
    # rate_list = rate["rate"]
    # for index, ele in enumerate(str_list):
    #     print(ele)
    #     ele_list = ele.strip().split(" ")
    #     ele_list = [int(element) for element in ele_list]
    #     record_diff_1 = np.diff(np.array(ele_list))
    #     print(record_diff_1)
    #     record_diff_2 = np.diff(record_diff_1)
    #     print(record_diff_2)
    #     print(rate_list[index])
    print("ok")


