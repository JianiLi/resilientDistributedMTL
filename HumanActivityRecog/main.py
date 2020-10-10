import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from agent import agent

torch.manual_seed(1)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


def value_to_tensor(values):
    device = get_device()
    return torch.from_numpy(values).float().to(device)


def readData():
    features = pd.read_csv('./UCI HAR Dataset/features.txt', sep='\s+', index_col=0, header=None)
    train_data = pd.read_csv('./UCI HAR Dataset/train/X_train.txt', sep='\s+',
                             names=list(features.values.ravel()))
    test_data = pd.read_csv('./UCI HAR Dataset/test/X_test.txt', sep='\s+',
                            names=list(features.values.ravel()))

    train_label = pd.read_csv('./UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None)
    test_label = pd.read_csv('./UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None)

    train_subject = pd.read_csv('./UCI HAR Dataset/train/subject_train.txt', sep='\s+', header=None)
    test_subject = pd.read_csv('./UCI HAR Dataset/test/subject_test.txt', sep='\s+', header=None)

    label_name = pd.read_csv('UCI HAR Dataset/activity_labels.txt', sep='\s+', header=None, index_col=0)

    train_data['label'] = train_label
    test_data['label'] = test_label

    train_data['subject'] = train_subject
    test_data['subject'] = test_subject

    def get_label_name(num):
        return label_name.iloc[num - 1, 0]

    train_data['label_name'] = train_data['label'].map(get_label_name)
    test_data['label_name'] = test_data['label'].map(get_label_name)

    # 原来标签为1-6，而算法需要0-5
    train_data['label'] = train_data['label'] - 1
    test_data['label'] = test_data['label'] - 1

    np.random.shuffle(train_data.values)
    np.random.shuffle(test_data.values)

    return train_data, test_data


def generateData(train_data, test_data, subject, batch_size):
    x_train = [d[:-3] for d in train_data.values if d[-2] == subject]
    y_train = [d[-3] for d in train_data.values if d[-2] == subject]
    x_test = [d[:-3] for d in test_data.values if d[-2] == subject]
    y_test = [d[-3] for d in test_data.values if d[-2] == subject]

    all_x_data = x_train + x_test
    all_y_data = y_train + y_test

    x_tensor = torch.FloatTensor(all_x_data)
    y_tensor = torch.LongTensor(all_y_data)

    all_data = []
    for i in range(len(x_tensor)):
        all_data.append([x_tensor[i], y_tensor[i]])

    np.random.shuffle(all_data)

    train_data_subject, val_data_subject, test_data_subject = all_data[:len(all_data) // 4 * 3], \
                                                              all_data[
                                                              len(all_data) // 4 * 3: len(all_data) // 8 * 7], all_data[
                                                                                                               len(
                                                                                                                   all_data) // 4 * 3:]

    # x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = x_tensor[:len(x_tensor) // 4 * 3], y_tensor[:len(x_tensor) // 4 * 3], \
    #                                    x_tensor[len(x_tensor) // 4 * 3:], y_tensor[len(x_tensor) // 4 * 3:]
    # x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor = x_test_tensor[:len(x_test_tensor) // 2], y_test_tensor[:len(x_test_tensor) // 2], \
    #                                     x_test_tensor[len(x_test_tensor) // 2:], y_test_tensor[len(x_test_tensor) // 2:]

    # train_data_subject = []
    # for i in range(len(x_train_tensor)):
    #     train_data_subject.append([x_train_tensor[i], y_train_tensor[i]])
    #
    # val_data_subject = []
    # for i in range(len(x_val_tensor)):
    #    val_data_subject.append([x_val_tensor[i], y_val_tensor[i]])
    #
    # test_data_subject = []
    # for i in range(len(x_test_tensor)):
    #    test_data_subject.append([x_test_tensor[i], y_test_tensor[i]])

    # un-even data
    # if subject % 5 == 0:
    #     train_data_subject, val_data_subject, test_data_subject = train_data_subject[:len(train_data_subject)//10], \
    #                                     val_data_subject[:len(val_data_subject)//10], test_data_subject[:len(test_data_subject)//10]

    train_loader_subject = DataLoader(dataset=train_data_subject, batch_size=batch_size, shuffle=True)
    test_loader_subject = DataLoader(dataset=test_data_subject, batch_size=batch_size, shuffle=True)
    val_loader_subject = DataLoader(dataset=val_data_subject, batch_size=batch_size, shuffle=True)

    return train_loader_subject, val_loader_subject, test_loader_subject


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # hidden layer
        self.out = torch.nn.Linear(n_hidden1, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = self.out(x)
        return x


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def getWeight(ego_k, A, Para_ex, Para_last, batch_x, batch_y, Accumulated_Loss, rule):
    Weight = []
    # gamma = 1 #gamma=1 represent using loss only
    gamma = 0.001

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
            if not l == ego_k:
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

    for k in range(0, N):
        Parameters_last.append({})
        Parameters_exchange.append({})
        a_last = A_last[k]
        a = A[k]
        for name, param in a.net.named_parameters():
            if param.requires_grad:
                if k in attacker:
                    # a.net.named_parameters()[name] = param.data * random.random() * 0.1
                    # original attack
                    # Parameters_exchange[k][name] = param.data * random.random() * 0.1
                    # attack with small perturbation (by reviewer 4)
                    Parameters_exchange[k][name] = param.data + random.random() * 0  # 1e-6

                else:
                    Parameters_exchange[k][name] = param.data
        for name, param in a_last.net.named_parameters():
            if param.requires_grad:
                if k in attacker:
                    # a_last.net.named_parameters()[name] = param.data * random.random() * 0.1
                    # original attack
                    # Parameters_last[k][name] = param.data * random.random() * 0.1
                    # attack with small perturbation (by reviewer 4)
                    Parameters_last[k][name] = param.data + random.random() * 0  # 1e-6
                else:
                    Parameters_last[k][name] = param.data

    Para_ex = []
    Para_last = []
    for k in range(0, N):
        para_ex_k = np.hstack(np.array([np.hstack(v.tolist()) for v in Parameters_exchange[k].values()]))
        para_last_k = np.hstack(np.array([np.hstack(v.tolist()) for v in Parameters_last[k].values()]))
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


def gaussian(x, mean, stddev):
    noise = Variable(x.new(x.size()).normal_(mean, stddev))
    return x + noise


def run(rule, attacker, epochs):
    torch.manual_seed(0)

    start_time = time.time()

    N = 30
    A = []
    batch_size = 10
    train_data, test_data = readData()

    Parameters = []

    attacker_num = len(attacker)

    # Accumulated_Loss = np.zeros((N, N))
    Accumulated_Loss = np.ones((N, N))
    middle1_neurons = 50

    Train_loader, Test_loader = [], []
    Val_loader_iter = []
    Val_loader = []

    average_train_loss, average_train_acc = [], []
    average_test_loss, average_test_acc = [], []

    individual_average_train_loss, individual_average_train_acc = np.zeros((epochs, N)), np.zeros((epochs, N))
    individual_average_test_loss, individual_average_test_acc = np.zeros((epochs, N)), np.zeros((epochs, N))

    for k in range(0, N):
        # net = Net(n_feature=561, n_hidden1=middle1_neurons, n_output=6)
        net = linearRegression(561, 6)
        a = agent(net)
        A.append(a)

        train_loader_no, val_loader_no, test_loader_no = generateData(train_data, test_data, k + 1, batch_size)
        Train_loader.append(train_loader_no)
        Test_loader.append(test_loader_no)
        Val_loader.append(val_loader_no)
        Val_loader_iter.append(iter(val_loader_no))

    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        Train_loader_iter = []
        total_train_loss = 0.
        total_train_acc = 0.
        total_eval_loss = 0.
        total_eval_acc = 0.

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
            Train_loader_iter.append(iter(Train_loader[k]))

        try:
            while True:
                A_last = deepcopy(A)
                Batch_X, Batch_Y = {}, {}
                for k in range(0, N):
                    # if k in attacker:
                    #     continue
                    batch_x, batch_y = Train_loader_iter[k].next()
                    Batch_X[k] = batch_x
                    Batch_Y[k] = batch_y
                    # only process 1/10 data for 1/3 of agents
                    if k % 3 == 0:
                        if random.randint(0, 10) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                            continue

                    # if k % 3 == 0:
                    #     train_loader = Train_loader_iter[k].next()
                    #     batch_x, batch_y = (train_loader[0]).narrow(0,0,1), (train_loader[1]).narrow(0,0,1)
                    # else:
                    #     batch_x, batch_y = Train_loader_iter[k].next()
                    #
                    # Batch_X.append(batch_x)
                    # Batch_Y.append(batch_y)

                    # if k % 3 == 0:
                    #     if random.randint(0, 5) == 1:
                    #         pass
                    # batch_x = gaussian(batch_x, 5, 5)
                    # batch_y = torch.LongTensor(np.random.randint(6, size=batch_size))
                    # if random.randint(0, 2) == 1:
                    #     batch_y = torch.LongTensor(np.random.randint(6, size=batch_size))
                    # if (k+1) % 5 == 0:
                    #     try:
                    #         batch_x, batch_y = Train_loader_iter[k].next()
                    #     except:
                    #         Train_loader_iter[k] = iter(Train_loader[k])
                    #         batch_x, batch_y = Train_loader_iter[k].next()
                    # else:
                    #     batch_x, batch_y = Train_loader_iter[k].next()
                    a = A[k]
                    loss, acc = a.optimize(batch_x, batch_y)
                    if math.isnan(loss) or math.isnan(acc):
                        continue
                    total_train_acc += acc
                    # try:
                    #     val_x, val_y = Val_loader_iter[k].next()
                    # except:
                    #     Val_loader_iter[k] = iter(Val_loader[k])
                    #     val_x, val_y = Val_loader_iter[k].next()
                    # Batch_X.append(val_x)
                    # Batch_Y.append(val_y)
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
                    batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
                    out = A[k].net(batch_x)
                    loss_func = torch.nn.CrossEntropyLoss()
                    loss = loss_func(out, batch_y)
                    pred = torch.max(out, 1)[1]
                    num_correct = (pred == batch_y).sum()
                    if math.isnan(loss) or math.isnan(num_correct):
                        continue
                    eval_loss += loss.item()
                    eval_acc += num_correct.item()
                    total_eval_loss += loss.item()
                    total_eval_acc += num_correct.item()
                    Eval_count[k] += len(batch_x)

                if not (math.isnan(eval_loss / Eval_count[k]) or math.isnan(eval_acc / Eval_count[k])):
                    ave_eval_loss += eval_loss / Eval_count[k]
                    ave_eval_acc += eval_acc / Eval_count[k]
                print('Agent: {:d}, Test Loss: {:.6f}, Acc: {:.6f}'.format(k, eval_loss / Eval_count[k],
                                                                           eval_acc / Eval_count[k]))
                individual_average_test_loss[epoch, k] = eval_loss / Eval_count[k]
                individual_average_test_acc[epoch, k] = eval_acc / Eval_count[k]

        try:
            print('Total Average Train Loss: {:.6f}, Train Acc: {:.6f}'.format(
                ave_train_loss / (N - nanCount - attacker_num), ave_train_acc / (N - nanCount - attacker_num)))
            average_train_loss.append(ave_train_loss / (N - nanCount - attacker_num))
            average_train_acc.append(ave_train_acc / (N - nanCount - attacker_num))
            print('Total Average Test Loss: {:.6f}, Test Acc: {:.6f}'.format(ave_eval_loss / (N - attacker_num),
                                                                             ave_eval_acc / (N - attacker_num)))
        except:
            pass

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

    epochs = 50
    N = 30
    # for attacker_num in [0, 10, 29]:
    attacker_num = 10
    random.seed(0)
    if attacker_num == 29:
        normal = [4]
        attacker = np.delete(range(0, 30), normal)
    else:
        attacker = random.sample(range(N), attacker_num)
    for rule in ["loss", "distance", "no-cooperation", "average"]:
        # for rule in ["loss", "distance"]:
        print("Total agent number:", N, "Attacker num:", attacker_num)
        print("attacker list:", attacker)
        print("rule: ", rule)
        run(rule, attacker, epochs)

