from torch.utils.data import WeightedRandomSampler, DataLoader

from model import *
from util import category_from_output, CustomTraceDataset, my_collate

n_hidden = 128
n_categories = 3
n_letters = 11


rnn = RNN(n_letters, n_hidden, n_categories)
rnn = torch.load('char-rnn-classification_saw.pt')
rnn.eval()
loss_fn = nn.CrossEntropyLoss()


# def evaluate(line_tensor):
#     hidden = rnn.init_hidden()
#
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
#
#     return output


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, total_acc, total_count = 0, 0, 0, 0
    with torch.no_grad():
        for batch, (records_batch, scores_batch) in enumerate(dataloader):
            idx_batch = 0
            for record in records_batch:
                hidden = model.init_hidden()
                for i in range(record.size()[0]):
                    output, hidden = rnn(record[i], hidden)  # tensor：(1,3)
                target = scores_batch[idx_batch]  # tensor：(1,) 具体是0,1,2
                test_loss += loss_fn(output, target)
                int_target = target.item() + 1
                total_acc = total_acc + 1 if category_from_output(output) == int_target else total_acc
                total_count = total_count + 1
                idx_batch = idx_batch + 1
    test_loss /= num_batches
    correct = total_acc / total_count
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# with torch.no_grad():
#     predictions = []
#     num_correct = 0
#     for input_line in time_records:
#         output = evaluate(line_to_tensor(input_line))
#         print(output)
#         top_n, top_i = output.topk(1)
#         category = top_i[0].item() + 1
#         predictions.append(category)
#     for i in range(0, len(time_records)):
#         if scores[i] == predictions[i]:
#             num_correct = num_correct+1
#     print(num_correct/len(time_records))
testing_data = CustomTraceDataset('/home/hadoop/PycharmProjects/Prediction_vertion1/data/prediction_version1.csv')
testing_dataloader = DataLoader(testing_data, batch_size=16, collate_fn=my_collate)
test_loop(testing_dataloader, rnn, loss_fn)
