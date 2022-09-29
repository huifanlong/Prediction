# 十一位二进制数足够保存2的11次方-1，即2047大小的数字。我们的数字代表的是视频的秒数，即可以保存长度高达2047秒（34min）的视频
import torch
# from util import *
import pandas as pd

data = pd.read_csv('/home/hadoop/PycharmProjects/Prediction_vertion1/data/prediction_version1.csv')
data = data[[len(data['str'][index].strip().split(" ")) > 30 for index in range(0, len(data))]]
# data = data[lis]
data.reset_index(drop=True, inplace=True)
print(data)

# # 创建DataFrame
# df = pd.DataFrame([['AAA'], ['BBB'], ['CCC'], [123]])
# # 删除含某特定字符串的行
# df1 = df.drop(df[df[0].str.contains('A', na=False)], inplace=True)
# # 删除含某特定数字的行
# df2 = df.drop(df[df[0] == 123].index, inplace=True)
print("ok")
print("done")





