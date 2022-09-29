from dataProcessing import read3, event
import pandas as pd

if __name__ == '__main__':
    # 进行event的encode
    # read3("data/data_delete_short.csv")
    #
    # name = ['event', 'position', 'timestamp', 'status', 'rate']
    # event_frame_list = [pd.DataFrame(columns=name, data=event_i) for event_i in event]
    # keys = [str(event_i) for event_i in range(0, len(event_frame_list))]
    # event_frame = pd.concat(event_frame_list, keys=keys)
    # event_frame.to_csv("data/data_delete_short.csv", encoding='gbk')

    # 在训练过程中对存储event的文件进行解析
    data = pd.read_csv("data/event_coding.csv")
    # print(data.iloc[len(data)-1, 0])  # 获取第一列的最大数(即uvPair数目)
    # print(data[data['Unnamed: 0'] == 0].iloc[:, 2:])  # 获取uvPair为0的event序列
    event_frame_list = [data[data['Unnamed: 0'] == event_i].iloc[:, 2:] for event_i in range(0, data.iloc[len(data)-1, 0])]
    print("ok")

