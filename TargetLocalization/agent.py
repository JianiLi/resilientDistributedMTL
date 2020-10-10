import torch
from torch.autograd import Variable

class agent:
    def __init__(self, net):
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_loss = 0
        self.train_acc = 0


    def optimize(self, batch_x, batch_y):
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = self.net(batch_x)
        loss = self.loss_func(out, batch_y)
        self.train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
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