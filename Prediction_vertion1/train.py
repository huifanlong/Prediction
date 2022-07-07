import torch
from data import *
from dataProcessing import *
from model import *
import random
import time
import math

n_hidden = 128
n_categories = 3
n_epochs = 1000
print_every = 50
plot_every = 10
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
all_categories = ["1", "2", "3"]


def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()+1
    return category_i


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example():
    index = random.randint(0, len(time_records) - 1)
    # print(index)
    category = scores[index]
    line = time_records[index]
    # print(category)
    # print(line)
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)  # tensor([2])?
    line_tensor = line_to_tensor(line)
    # print(category_tensor)
    # print(line_tensor)
    return category, line, category_tensor, line_tensor


rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()


# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()
read3()

correct_num = 0
total_num = 0
guess_3 = 0
guess_2 = 0
for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    guess = category_from_output(output)
    if guess == int(category):
        correct_num = correct_num+1
    if guess == 3:
        guess_3 = guess_3 + 1
    elif guess == 2:
        guess_2 = guess_2 + 1
    total_num = total_num + 1
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess = category_from_output(output)
        correct = '✓' if guess == int(category) else '✗ (%s)' % category
        # print('%d %d%% (%s) %.4f %s / %s %s' % (
        # epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))
        print('%d %d%% (%s) %.4f %s / %s %s %s %s' % (
            epoch, epoch / n_epochs * 100, timeSince(start), loss, correct_num/total_num, guess, correct, guess_2/total_num, guess_3/total_num))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification2.pt')
