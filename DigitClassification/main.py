import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from agent import agent

# from torchsummary import summary

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def readData_mnist():
    train_data = torchvision.datasets.MNIST(
        './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
    )
    test_data = torchvision.datasets.MNIST(
        './mnist', train=False, transform=torchvision.transforms.ToTensor()
    )
    print("train_data:", train_data.train_data.size())
    print("train_labels:", train_data.train_labels.size())
    print("test_data:", test_data.test_data.size())
    return train_data, test_data


def generateData_mnist(train_data, test_data, tr_split_len, te_split_len, number):
    # if number == 1:
    #     train_x_no = train_data.train_data[number*tr_split_len : number*tr_split_len + tr_split_len//10].float()/255
    #     train_x_no = train_x_no.repeat(10, 1, 1)
    #     train_y_no = train_data.train_labels[number*tr_split_len : number*tr_split_len + tr_split_len//10]
    #     train_y_no = train_y_no.repeat(10)
    # else:
    train_x_no = train_data.train_data[number * tr_split_len: (number + 1) * tr_split_len].float() / 255
    train_y_no = train_data.train_labels[number * tr_split_len: (number + 1) * tr_split_len]
    test_x_no = test_data.test_data[number * te_split_len: (number + 1) * te_split_len].float() / 255
    test_y_no = test_data.test_labels[number * te_split_len: (number + 1) * te_split_len]

    train_data_no = []
    if number == 1:
        # wrong label / uneven data
        for i in range(len(train_x_no)):
            train_data_no.append([train_x_no[i].unsqueeze(0), train_y_no[i]])
    else:
        for i in range(len(train_x_no)):
            train_data_no.append([train_x_no[i].unsqueeze(0), train_y_no[i]])

    # print(train_x_1[i].float()/255)
    test_data_no = []
    for i in range(len(test_x_no)):
        test_data_no.append([test_x_no[i].unsqueeze(0), test_y_no[i]])

    train_loader_no = DataLoader(dataset=train_data_no, batch_size=64, shuffle=True)
    test_loader_no = DataLoader(dataset=test_data_no, batch_size=64)

    return train_loader_no, test_loader_no


def readData_synthetic_digits():
    transformtrain = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    transformtest = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = torchvision.datasets.ImageFolder('synthetic_digits/synthetic_digits/imgs_train',
                                                  transform=transformtrain)
    test_data = torchvision.datasets.ImageFolder('synthetic_digits/synthetic_digits/imgs_valid',
                                                 transform=transformtest)

    # np.random.shuffle(train_data)
    # shuffled in ImageFolder

    return train_data, test_data


def generateData_synthetic_digits(remaining_tr, remaining_te, tr_split_len, te_split_len):
    part_tr, part_tr2 = torch.utils.data.random_split(remaining_tr, [tr_split_len, len(remaining_tr) - tr_split_len])
    part_te, part_te2 = torch.utils.data.random_split(remaining_te, [te_split_len, len(remaining_te) - te_split_len])

    train_loader_no = DataLoader(part_tr, batch_size=128, shuffle=True)
    test_loader_no = DataLoader(part_te, batch_size=128, shuffle=False)

    return train_loader_no, test_loader_no, part_tr2, part_te2


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


def getWeight(ego_k, A, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule):
    Weight = []
    gamma = 0.05
    N = len(A)

    if rule == "loss":
        #     # loss based filtering 1/0
        #     for l in range(0, N):
        #         loss = A[k].getLoss(batch_x, batch_y, A[l].net)
        #         Accumulated_Loss[ego_k, l] = (1 - gamma) * Accumulated_Loss[ego_k, l] + gamma * loss
        #     for l in range(0, N):
        #         weight = 1 if l == np.argmin(Accumulated_Loss[ego_k, :]) else 0
        #         Weight.append(weight)
        # elif rule == "reversedLoss":
        # Reversed loss
        Weight = np.zeros((N,))
        reversed_Loss = np.zeros((N,))
        loss = A[ego_k].getLoss(batch_x, batch_y, A[ego_k].net)
        Accumulated_Loss[ego_k, ego_k] = (1 - gamma) * Accumulated_Loss[ego_k, ego_k] + gamma * loss
        for l in range(0, N):
            if not l == ego_k:
                loss = A[ego_k].getLoss(batch_x, batch_y, A[l].net)
                Accumulated_Loss[ego_k, l] = (1 - gamma) * Accumulated_Loss[ego_k, l] + gamma * loss
            if Accumulated_Loss[ego_k, l] <= Accumulated_Loss[ego_k, ego_k]:
                reversed_Loss[l] = 1. / Accumulated_Loss[ego_k, l]
        sum_reversedLoss = sum(reversed_Loss)
        for l in range(0, N):
            if Accumulated_Loss[ego_k, l] <= Accumulated_Loss[ego_k, ego_k]:
                weight = reversed_Loss[l] / sum_reversedLoss
                Weight[l] = weight
    elif rule == "distance":
        Weight = np.zeros((N,))
        reversed_Loss = np.zeros((N,))
        para_ex_k = Para_ex[ego_k]
        para_last_k = Para_last[ego_k]
        dist = np.linalg.norm(para_ex_k - para_last_k)
        Accumulated_Loss[ego_k, ego_k] = (1 - gamma) * Accumulated_Loss[ego_k, ego_k] + gamma * dist ** 2
        for l in range(0, N):
            para_ex_l = Para_ex[l]
            para_last_k = Para_last[ego_k]
            # print(np.linalg.norm(para_ex_l - para_last_k))
            dist = np.linalg.norm(para_ex_l - para_last_k)
            Accumulated_Loss[ego_k, l] = (1 - gamma) * Accumulated_Loss[ego_k, l] + gamma * dist ** 2
            # if Accumulated_Loss[ego_k,l] <= Accumulated_Loss[ego_k, ego_k]:
            reversed_Loss[l] = 1. / Accumulated_Loss[ego_k, l]
        sum_reversedLoss = sum(reversed_Loss)
        for l in range(0, N):
            # if Accumulated_Loss[ego_k,l] <= Accumulated_Loss[ego_k, ego_k]:
            weight = reversed_Loss[l] / sum_reversedLoss
            Weight[l] = weight
    elif rule == "average":
        # average based weight
        for l in range(0, N):
            if not l == ego_k:
                weight = 1 / N
            else:
                weight = 1 - (N - 1) / N
            Weight.append(weight)
    elif rule == "no-cooperation":
        for l in range(0, N):
            if l == ego_k:
                weight = 1
            else:
                weight = 0
            Weight.append(weight)
    else:
        return Weight, Accumulated_Loss

    return Weight, Accumulated_Loss


def cooperation(A, A_last, Batch_X, Batch_Y, Accumulated_Loss, rule, attacker):
    Parameters_last = []
    Parameters_exchange = []
    N = len(A)

    for k in range(0, N):
        Parameters_last.append({})
        Parameters_exchange.append({})
        a_last = A_last[k]
        a = A[k]
        for name, param in a.net.named_parameters():
            if param.requires_grad:
                if k in attacker:
                    # a.net.named_parameters()[name] = param.data * random.random() * 0.1
                    Parameters_exchange[k][name] = param.data * random.random() * 0.1
                else:
                    Parameters_exchange[k][name] = param.data
        for name, param in a_last.net.named_parameters():
            if param.requires_grad:
                if k in attacker:
                    # a_last.net.named_parameters()[name] = param.data * random.random() * 0.1
                    Parameters_last[k][name] = param.data * random.random() * 0.1
                else:
                    Parameters_last[k][name] = param.data

    Para_ex = []
    Para_last = []
    for k in range(0, N):
        para_ex_k = np.hstack([v.flatten().tolist() for v in Parameters_exchange[k].values()])
        para_last_k = np.hstack([v.flatten().tolist() for v in Parameters_last[k].values()])
        Para_ex.append(para_ex_k)
        Para_last.append(para_last_k)

    Parameters = deepcopy(Parameters_exchange)
    for k in range(0, N):
        a = A[k]
        if k not in attacker:
            batch_x, batch_y = Batch_X[k], Batch_Y[k]
            Weight, Accumulated_Loss = getWeight(k, A, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule)
            # print(Accumulated_Loss)
            # print(Weight)

            for name, param in a.net.named_parameters():
                Parameters[k][name] = 0. * Parameters[k][name]
                for l in range(0, N):
                    if param.requires_grad:
                        Parameters[k][name] += Parameters_exchange[l][name] * Weight[l]

    for k in range(0, N):
        a = A[k]
        for name, param in a.net.named_parameters():
            param.data = Parameters[k][name]

    return A, Accumulated_Loss


def run(rule, attacker, epochs):
    torch.manual_seed(1)

    start_time = time.time()

    N = 10
    N1 = 5
    tr_split_len1 = 2000
    te_split_len1 = 400
    tr_split_len2 = 2000
    te_split_len2 = 400
    A = []
    train_data1, test_data1 = readData_mnist()
    train_data2, test_data2 = readData_synthetic_digits()
    remaining_tr, remaining_te = train_data2, test_data2

    Parameters = []

    # attacker_num = 2
    # attacker = [2, 7]

    attacker_num = len(attacker)
    # Accumulated_Loss = np.zeros((N, N))
    Accumulated_Loss = np.ones((N, N))

    average_train_loss, average_train_acc = [], []
    average_test_loss, average_test_acc = [], []

    individual_average_train_loss, individual_average_train_acc = np.zeros((epochs, N)), np.zeros((epochs, N))
    individual_average_test_loss, individual_average_test_acc = np.zeros((epochs, N)), np.zeros((epochs, N))

    for k in range(0, N):
        net = Net().to(device)
        # print(net)
        # summary(net, (1,28,28), batch_size=-1)
        a = agent(net)
        A.append(a)
        Parameters.append({})

        for name, param in a.net.named_parameters():
            if param.requires_grad:
                Parameters[k][name] = param.data

    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        Train_loader_iter = []
        Test_loader = []
        total_train_loss = 0.
        total_train_acc = 0.
        total_eval_loss = 0.
        total_eval_acc = 0.
        remaining_tr, remaining_te = train_data2, test_data2

        Count = np.zeros((N,))

        ave_train_loss = 0.
        ave_train_acc = 0.
        ave_eval_loss = 0.
        ave_eval_acc = 0.
        nanCount = 0

        for k in range(0, N):
            a = A[k]
            a.train_loss = 0.
            a.train_acc = 0.

            if k < N1:
                train_loader_no, test_loader_no = generateData_mnist(train_data1, test_data1, tr_split_len1,
                                                                     te_split_len1, k)
            else:
                train_loader_no, test_loader_no, remaining_tr, remaining_te = generateData_synthetic_digits(
                    remaining_tr,
                    remaining_te,
                    tr_split_len2,
                    te_split_len2)

            Train_loader_iter.append(iter(train_loader_no))
            Test_loader.append(test_loader_no)

        # for iteration in range(0, tr_split_len//64):
        # for k in range(0, N):
        # training-----------------------------
        try:
            while True:
                A_last = deepcopy(A)
                Batch_X, Batch_Y = {}, {}
                for k in range(0, N):
                    batch_x, batch_y = next(Train_loader_iter[k])
                    Batch_X[k] = batch_x.to(device)
                    Batch_Y[k] = batch_y.to(device)
                    if k in attacker:
                        continue
                    # 5 agents, get access to 1, 1/2, 1/3, 1/5, 1/10 data, so their models have different accuracy
                    if k % 5 == 0:
                        if random.randint(0, 1) in [0]:
                            continue
                    if k % 5 == 1:
                        if random.randint(0, 2) in [0, 1]:
                            continue
                    if k % 5 in [2, 3]:
                        if random.randint(0, 3) in [0, 1, 2]:
                            continue
                    # if k % 5 == 3:
                    #     if random.randint(0, 9) in [0,1,2,3,4,5,6,7,8]:
                    #         continue
                    a = A[k]
                    loss, acc = a.optimize(batch_x.to(device), batch_y.to(device))
                    total_train_loss += loss
                    total_train_acc += acc
                    Count[k] += len(batch_x)

                A, Accumulated_Loss = cooperation(A, A_last, Batch_X, Batch_Y, Accumulated_Loss, rule, attacker)
                # print(Accumulated_Loss)

        except StopIteration:
            # print(iteration)
            Eval_count = np.zeros((N,))
            for k in range(0, N):
                if k in attacker:
                    continue
                print('Agent: {:d}, Train Loss: {:.6f}, Acc: {:.6f}'.format(k, A[k].train_loss / Count[k],
                                                                            A[k].train_acc / Count[k]))
                individual_average_train_loss[epoch, k] = A[k].train_loss / Count[k]
                individual_average_train_acc[epoch, k] = A[k].train_acc / Count[k]

                if not (math.isnan(A[k].train_loss / Count[k]) or math.isnan(A[k].train_acc / Count[k])):
                    ave_train_loss += A[k].train_loss / Count[k]
                    ave_train_acc += A[k].train_acc / Count[k]
                else:
                    nanCount += 1

                # evaluation--------------------------------
                A[k].net.eval()
                eval_loss = 0.
                eval_acc = 0.
                for batch_x, batch_y in Test_loader[k]:
                    batch_x, batch_y = Variable(batch_x, volatile=True).to(device), Variable(batch_y, volatile=True).to(
                        device)
                    out = A[k].net(batch_x)
                    loss_func = torch.nn.CrossEntropyLoss()
                    loss = loss_func(out, batch_y)
                    eval_loss += loss.item()
                    total_eval_loss += loss.item()
                    pred = torch.max(out, 1)[1]
                    num_correct = (pred == batch_y).sum()
                    eval_acc += num_correct.item()
                    total_eval_acc += num_correct.item()
                    Eval_count[k] += len(batch_x)

                if not (math.isnan(eval_loss / Eval_count[k]) or math.isnan(eval_acc / Eval_count[k])):
                    ave_eval_loss += eval_loss / Eval_count[k]
                    ave_eval_acc += eval_acc / Eval_count[k]
                print('Agent: {:d}, Test Loss: {:.6f}, Acc: {:.6f}'.format(k, eval_loss / Eval_count[k],
                                                                           eval_acc / Eval_count[k]))
                individual_average_test_loss[epoch, k] = eval_loss / Eval_count[k]
                individual_average_test_acc[epoch, k] = eval_acc / Eval_count[k]

        # print('Total Average Train Loss: {:.6f}, Train Acc: {:.6f}'.format(total_train_loss / sum(Count),
        #                                                                    total_train_acc / sum(Count)))
        # average_train_loss.append(total_train_loss / sum(Count))
        # average_train_acc.append(total_train_acc / sum(Count))
        # print('Total Average Test Loss: {:.6f}, Test Acc: {:.6f}'.format(total_eval_loss / sum(Eval_count),
        #                                                                  total_eval_acc / sum(Eval_count)))
        #
        # print('Training time by far: {:.2f}s'.format(time.time() - start_time))
        # average_test_loss.append(total_eval_loss / sum(Eval_count))
        # average_test_acc.append(total_eval_acc / sum(Eval_count))

        print(
            'Total Average Train Loss: {:.6f}, Train Acc: {:.6f}'.format(ave_train_loss / (N - nanCount - attacker_num),
                                                                         ave_train_acc / (N - nanCount - attacker_num)))
        average_train_loss.append(ave_train_loss / (N - nanCount - attacker_num))
        average_train_acc.append(ave_train_acc / (N - nanCount - attacker_num))
        print('Total Average Test Loss: {:.6f}, Test Acc: {:.6f}'.format(ave_eval_loss / (N - attacker_num),
                                                                         ave_eval_acc / (N - attacker_num)))

        print('Training time by far: {:.2f}s'.format(time.time() - start_time))
        average_test_loss.append(ave_eval_loss / (N - attacker_num))
        average_test_acc.append(ave_eval_acc / (N - attacker_num))

        if epoch % 10 == 0 or epoch == epochs - 1:
            if attacker_num == 0:
                try:
                    os.makedirs("results")
                except OSError:
                    print("Creation of the directory %s failed")
                np.save('results/average_train_loss_%s.npy' % rule, average_train_loss)
                np.save('results/average_train_acc_%s.npy' % rule, average_train_acc)
                np.save('results/average_test_loss_%s.npy' % rule, average_test_loss)
                np.save('results/average_test_acc_%s.npy' % rule, average_test_acc)
                np.save('results/individual_average_train_loss_%s.npy' % rule, individual_average_train_loss)
                np.save('results/individual_average_train_acc_%s.npy' % rule, individual_average_train_acc)
                np.save('results/individual_average_test_loss_%s.npy' % rule, individual_average_test_loss)
                np.save('results/individual_average_test_acc_%s.npy' % rule, individual_average_test_acc)
            else:
                try:
                    os.makedirs("results/attacked/%d" % attacker_num)
                except OSError:
                    print("Creation of the directory %s failed")
                np.save('results/attacked/%d/average_train_loss_%s.npy' % (attacker_num, rule), average_train_loss)
                np.save('results/attacked/%d/average_train_acc_%s.npy' % (attacker_num, rule), average_train_acc)
                np.save('results/attacked/%d/average_test_loss_%s.npy' % (attacker_num, rule), average_test_loss)
                np.save('results/attacked/%d/average_test_acc_%s.npy' % (attacker_num, rule), average_test_acc)
                np.save('results/attacked/%d/individual_average_train_loss_%s.npy' % (attacker_num, rule),
                        individual_average_train_loss)
                np.save('results/attacked/%d/individual_average_train_acc_%s.npy' % (attacker_num, rule),
                        individual_average_train_acc)
                np.save('results/attacked/%d/individual_average_test_loss_%s.npy' % (attacker_num, rule),
                        individual_average_test_loss)
                np.save('results/attacked/%d/individual_average_test_acc_%s.npy' % (attacker_num, rule),
                        individual_average_test_acc)


if __name__ == '__main__':
    epochs = 100
    # for rule in ["no-cooperation", "loss", "distance", " average"]:
    for rule in ["loss", "distance"]:
        # for attacker in [[0,1,2,3,5,6,7,8],[], [2, 7]]:
        for attacker in [[], [2, 7]]:
            run(rule, attacker, epochs)
