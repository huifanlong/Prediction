from dataProcessing import *
from model import *

n_hidden = 128
n_categories = 3
n_letters = 11

read3()

rnn = RNN(n_letters, n_hidden, n_categories)
rnn = torch.load('char-rnn-classification.pt')
rnn.eval()


def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


with torch.no_grad():
    predictions = []
    num_correct = 0
    for input_line in time_records:
        output = evaluate(line_to_tensor(input_line))
        top_n, top_i = output.topk(1)
        category = top_i[0].item() + 1
        predictions.append(category)
    for i in range(0, len(time_records)):
        if scores[i] == predictions[i]:
            num_correct = num_correct+1
    print(num_correct/len(time_records))