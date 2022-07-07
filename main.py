from dataProcessing import *
from model import *
from train import scores
import random

if __name__ == '__main__':
    read3()
    num_1, num_2, num_3 = 0, 0, 0
    for i in range(0, len(scores)):
        if int(scores[i]) == 1:
            num_1 = num_1+1
        elif int(scores[i]) == 2:
            num_2 = num_2 + 1
        else:
            num_3 = num_3 + 1
    total = num_1+num_2+num_3
    print(num_1/total)
    print(num_2/total)
    print(num_3/total)