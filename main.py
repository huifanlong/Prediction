from dataProcessing import read3, event
import pandas as pd

if __name__ == '__main__':
    read3("data/data_delete_short.csv")

    name = ['event', 'position', 'timestamp', 'status', 'rate']
    event_frame_list = [pd.DataFrame(columns=name, data=event_i) for event_i in event]
    keys = [str(event_i) for event_i in range(0, len(event_frame_list))]
    event_frame = pd.concat(event_frame_list, keys=keys)
    event_frame.to_csv("data/data_delete_short.csv", encoding='gbk')

    # data = pd.read_csv("data/event_coding.csv")
    # time_records1 = data[['one']]
    # print(time_records1)
