# import torch
#
# class agent:
#     def __init__(self, net):
#         self.net = net
#
#
#     def optimizer(self, x, y):
#         # print(net)  # net architecture
#
#         optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
#         loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
#
#         #plt.ion()  # something about plotting
#
#         for t in range(10):
#             out = self.net(x)  # input x and predict based on x
#             loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
#             optimizer.zero_grad()  # clear gradients for next train
#             loss.backward()  # backpropagation, compute gradients
#             optimizer.step()  # apply gradients
#
#         prediction = torch.max(out, 1)[1]
#         pred_y = prediction.data.numpy()
#         target_y = y.data.numpy()
#         accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#
#         return accuracy, loss.item()

import torch
from torch.autograd import Variable
import math

class agent:
    def __init__(self, net):
        self.net = net
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_loss = 0
        self.train_acc = 0


    def optimize(self, batch_x, batch_y):
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = self.net(batch_x)
        loss = self.loss_func(out, batch_y)
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()

        if math.isnan(loss.item()) or math.isnan(train_correct.item()):
            return loss.item(), train_correct.item()

        self.train_loss += loss.item()
        self.train_acc += train_correct.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_correct.item()


    def getLoss(self, batch_x, batch_y, neighbor_net):
        neighbor_net.eval()
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = neighbor_net(batch_x)
        loss = self.loss_func(out, batch_y)

        return loss.item()