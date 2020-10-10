import copy
import random
import numpy as np
from matplotlib import rc
from scipy.spatial.distance import cdist, euclidean
from utils import *
from collections import defaultdict


def noncooperative_learn(i, numAgents, w0, x, vu, u, vd, d, psi, w, mu_k, q, attackers, psi_a):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        u[:, k] = unit + vu[i, k]
        d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]
    a = 0
    for k in range(numAgents):
        if k not in attackers:
            # target estimation
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])
            w[:, k] = psi[:, k]
        else:
            w[:, k] = psi_a[:, a]
            a += 1


    return w


def average_learn(i, numAgents, w0, x, vu, u, vd, d, psi, w, mu_k, q, attackers, psi_a, Neigh):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        u[:, k] = unit +  vu[i, k]
        d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]

    a = 0
    for k in range(numAgents):
        if k not in attackers:
            # target estimation
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])

        else:
            psi[:, k] = psi_a[:, a]
            a += 1

    for k in range(numAgents):
        if k not in attackers:
            w[:, k] = np.mean(np.array([psi[:, j] for j in Neigh[k]]), axis=0)
        else:
            w[:, k] = psi[:, k]

    return w

def loss_learn(i, numAgents, w0, x, vu, u, vd, d, psi, w, mu_k, q, attackers, psi_a, Accumulated_Loss,  Neigh):
    for k in range(numAgents):
        dist = w0.distance(x[k])
        unit = [(w0.x - x[k].x) / dist, (w0.y - x[k].y) / dist]
        u[:, k] = unit + vu[i, k]
        d[k] = np.dot([w0.x - x[k].x, w0.y - x[k].y], u[:, k].T) + vd[i, k]
        q[:, k] = [x[k].x, x[k].y] + d[k] * u[:, k]

    a = 0
    gamma = 0.01

    for k in range(numAgents):
        if k not in attackers:
            psi[:, k] = w[:, k] + mu_k * (q[:, k] - w[:, k])

        else:
            psi[:, k] = psi_a[:, a]
            a += 1


    for k in range(numAgents):
        if k not in attackers:
            Weight = np.zeros((numAgents,))
            reversed_loss = np.zeros((numAgents,))
            loss = (d[k] - (np.dot([psi[:, k]], u[:, k].T)).item()) ** 2
            Accumulated_Loss[k, k] = (1 - gamma) * Accumulated_Loss[k, k] + gamma * loss
            for l in Neigh[k]:
                loss = (d[k] - (np.dot([psi[:, l]], u[:, k].T)).item())**2
                Accumulated_Loss[k, l] = (1 - gamma) * Accumulated_Loss[k, l] + gamma * loss
                if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
                    reversed_loss[l] = (1./Accumulated_Loss[k, l])
            sum_reversedLoss = sum(reversed_loss)
            for l in Neigh[k]:
                if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
                    weight = reversed_loss[l] / sum_reversedLoss
                    Weight[l] = weight
            #print(Weight)
            w[:, k] = np.dot(psi, Weight)
        else:
            w[:, k] = psi[:, k]

    return w, Accumulated_Loss


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    # parameters
    iteration = 1000
    sensingRange = 1
    r = 1
    box = 12
    numAgents = 100
    mu_k = 0.01
    w0 = Point(10, 10)
    lower = 0
    upper = 3

    attackerNum = 5
    attackers = random.sample(list(range(numAgents)), attackerNum)
    normalAgents = [k for k in range(numAgents) if k not in attackers]

    x_no = random_point_set(numAgents, lower=lower, upper=upper)
    # for k in attackers:
    #     w_no[k] = Point(np.random.random(), np.random.random())

    x_init = copy.deepcopy(x_no)
    x_avg = copy.deepcopy(x_no)
    x_loss = copy.deepcopy(x_no)

    Neigh = []
    for k in range(numAgents):
        neighbor = findNeighbors(x_init, k, numAgents, sensingRange, maxNeighborSize=10)
        Neigh.append(neighbor)


    plt.clf()
    # plt.grid(True, which='major')
    ax = plt.gca()
    ax.set_xlim(lower-0.1, upper+0.1)
    ax.set_ylim(lower-0.1, upper+0.1)
    # plt.xticks([0.3*i for i in range(-5,5, 1)])
    # plt.yticks([0.3*i for i in range(-5,5, 1)])
    plt.gca().set_aspect('equal', adjustable='box')
    # ax.grid(True)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
        tic.label1On = tic.label2On = False
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end + 0.01, 0.2))
    ax.yaxis.set_ticks(np.arange(start, end + 0.01, 0.2))

    # ax.set_xticks([0, 0.3, 0.4, 1.0, 1.5])
    ax.set_xticklabels([-1, "", "", "", "", 0, "", "", "", "", 1])
    ax.set_yticklabels([-1, "", "", "", "", 0, "", "", "", "", 1])
    for i in range(0, numAgents):
        for neighbor in Neigh[i]:
            plt.plot([x_init[i].x, x_init[neighbor].x], [x_init[i].y, x_init[neighbor].y], linewidth=0.2,
                     color='gray')

    plot_point_set(x_init, color='b')  # fault-free robots are plotted in blue
    plot_point_set([x_init[p] for p in attackers], color='r')  # faulty robots are plotted in red

    plt.pause(0.1)
    # plt.show()
    # plt.savefig('./result/largeNetwork/%s%d.eps' % (method, t))
    # end = input('Press enter to end the program.')



    psi_a = 0 * np.ones((2, len(attackers)))
    phi_a = 0 * np.ones((2, len(attackers)))

    mu_vd = 0
    mu_vu = 0
    sigma_vd2 = 0.5 + 0.5 * np.random.random((numAgents, 1))
    #sigma_vd2[random.sample(range(numAgents), 20)] = 5
    sigma_vu2 = 0.01 + 0.04 * np.random.random((numAgents, 1))
    sigma_vu2[random.sample(range(numAgents), 5)] = 0.3

    # The following parameters work
    # sigma_vd2 = 1 + 0.4 * np.random.random((numAgents, 1))
    # #sigma_vd2[random.sample(range(numAgents), 3)] = 3
    # sigma_vu2 = 0.5 + 0.05 * np.random.random((numAgents, 1))
    # #sigma_vu2[random.sample(range(numAgents), 3)] = 3
    vd = np.zeros((iteration, numAgents))
    vu = np.zeros((iteration, numAgents))
    for k in range(numAgents):
        vd[:, k] = np.random.normal(mu_vd, sigma_vd2[k], iteration)
        vu[:, k] = np.random.normal(mu_vu, sigma_vu2[k], iteration)

    d = np.zeros((numAgents,))
    u = np.zeros((2, numAgents))
    q = np.zeros((2, numAgents))
    psi = np.zeros((2, numAgents))

    w_no = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_no[0, k], w_no[1, k] = x_no[k].x, x_no[k].y
    w_avg = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_avg[0, k], w_avg[1, k] = x_avg[k].x, x_avg[k].y
    w_loss = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_loss[0, k], w_loss[1, k] = x_loss[k].x, x_loss[k].y

    vg_no = np.zeros((2, numAgents))
    vg_avg = np.zeros((2, numAgents))
    vg_loss = np.zeros((2, numAgents))


    phi = np.zeros((2, numAgents))
    v_no = np.zeros((2, numAgents))
    v_avg = np.zeros((2, numAgents))
    v_loss = np.zeros((2, numAgents))

    MSE_x_no = np.zeros((iteration,))
    MSE_x_avg = np.zeros((iteration,))
    MSE_x_loss = np.zeros((iteration,))

    W1_no = np.zeros((iteration, numAgents))
    W1_avg = np.zeros((iteration, numAgents))
    W1_loss = np.zeros((iteration, numAgents))

    Accumulated_Loss = np.zeros((numAgents, numAgents))

    fig = plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(151)
    ax2 = plt.subplot(152)
    ax3 = plt.subplot(153)


    for i in range(iteration):
        error_no = 0
        for k in normalAgents:
            agent = Point(w_no[0, k], w_no[1, k])
            error_no += agent.distance(w0) ** 2
            W1_no[i, k] = w_no[0, k]
        error_no /= len(w_no)
        MSE_x_no[i] = error_no

        error_avg = 0
        for k in normalAgents:
            agent = Point(w_avg[0, k], w_avg[1, k])
            error_avg += agent.distance(w0) ** 2
            W1_avg[i, k] = w_avg[0, k]
        error_avg /= len(w_avg)
        MSE_x_avg[i] = error_avg

        error_loss = 0
        for k in normalAgents:
            agent = Point(w_loss[0, k], w_loss[1, k])
            error_loss += (agent.distance(w0)) ** 2
            W1_loss[i, k] = w_loss[0, k]
        error_loss /= len(w_loss)
        MSE_x_loss[i] = error_loss

        print('iteration %d' % i)

        # plt.clf()

        # ax1 = plt.subplot(151)
        # ax1.set_xlim(-1, box)
        # ax1.set_ylim(-1, box)
        # ax1.set_aspect('equal', adjustable='box')
        # for tic in ax1.xaxis.get_major_ticks():
        #     tic.tick1On = tic.tick2On = False
        #     tic.label1On = tic.label2On = False
        # for tic in ax1.yaxis.get_major_ticks():
        #     tic.tick1On = tic.tick2On = False
        #     tic.label1On = tic.label2On = False
        # plot_point(w0, marker='*', color='chartreuse', size=12, ax=ax1)
        # plot_point_set(x_no, color='b', ax=ax1, alpha=0.5)
        # if attackers:
        #     for i in attackers:
        #         plot_point(x_no[i], color='r', ax=ax1)
        # ax1.set_title('Noncooperative LMS')
        #
        # ax2 = plt.subplot(152)
        # ax2.set_xlim(-1, box)
        # ax2.set_ylim(-1, box)
        # ax2.set_aspect('equal', adjustable='box')
        # for tic in ax2.xaxis.get_major_ticks():
        #     tic.tick1On = tic.tick2On = False
        #     tic.label1On = tic.label2On = False
        # for tic in ax2.yaxis.get_major_ticks():
        #     tic.tick1On = tic.tick2On = False
        #     tic.label1On = tic.label2On = False
        # plot_point(w0, marker='*', color='chartreuse', size=12, ax=ax2)
        # plot_point_set(x_avg, color='b', ax=ax2, alpha=0.5)
        # if attackers:
        #     for i in attackers:
        #         plot_point(x_avg[i], color='r', ax=ax2)
        # ax2.set_title('Average')
        #
        # ax3 = plt.subplot(153)
        # ax3.set_xlim(-1, box)
        # ax3.set_ylim(-1, box)
        # ax3.set_aspect('equal', adjustable='box')
        # for tic in ax3.xaxis.get_major_ticks():
        #     tic.tick1On = tic.tick2On = False
        #     tic.label1On = tic.label2On = False
        # for tic in ax3.yaxis.get_major_ticks():
        #     tic.tick1On = tic.tick2On = False
        #     tic.label1On = tic.label2On = False
        # plot_point(w0, marker='*', color='chartreuse', size=12, ax=ax3)
        # plot_point_set(x_loss, color='b', ax=ax3, alpha=0.5)
        # if attackers:
        #     for i in attackers:
        #         plot_point(x_loss[i], color='r', ax=ax3)
        # ax3.set_title('Coordinate-wise median')
        #
        #
        #
        # plt.pause(0.001)

        # noncooperative
        w_no = noncooperative_learn(i, numAgents, w0, x_no, vu, u, vd, d, psi, w_no, mu_k, q, attackers, psi_a)

        # cooperative
        w_avg = average_learn(i, numAgents, w0, x_avg, vu, u, vd, d, psi, w_avg, mu_k, q, attackers, psi_a, Neigh)

        w_loss, Accumulated_Loss = loss_learn(i, numAgents, w0, x_loss, vu, u, vd, d, psi, w_loss, mu_k, q, attackers, psi_a, Accumulated_Loss, Neigh)




    fig1 = plt.figure(figsize=(3.9, 2.5))
    #fig1 = plt.figure(figsize=(3.9, 2))
    plt.plot(10 * np.log10(MSE_x_no[1:]), label=r'Non-coop')
    plt.plot(10 * np.log10(MSE_x_avg[1:]), label=r'Average')
    plt.plot(10 * np.log10(MSE_x_loss[1:]), label=r'loss-based')
    # plt.plot(MSE_x_no[1:], label=r'Non-coop')
    # plt.plot((MSE_x_avg[1:]), label=r'Average')
    # plt.plot((MSE_x_loss[1:]), label=r'loss-based')


    # plt.title('cooperative under attack using median')
    plt.xlabel(r'iteration $i$', fontsize=10)
    plt.ylabel(r'MSD (dB)', fontsize=10)
    #plt.xticks([0, 100, 200, 300, 400, 500])
    # plt.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0.34, 0.43))
    # plt.yticks([-30,-15,0,15,30])
    # plt.legend(fontsize=7, loc='best')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.yticks([-75, -50, -25, 0, 25])
    # if attackerNum == 6:
    #     plt.yticks([-40, -20, 0, 20, 40])
    #     plt.ylim([-45, 45])
    # elif attackerNum == 0:
    #     plt.yticks([-60, -40, -20, 0, 20])
    #     plt.ylim([-70, 40])
    plt.tight_layout()
    plt.show()
    #fig1.savefig('fig/MSD_mobile_attack%d.eps' % attackerNum)

    plt.subplot(151)
    plt.plot(10 * np.log10(MSE_x_no))
    plt.title('Noncooperative LMS')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')

    plt.subplot(152)
    plt.plot(10 * np.log10(MSE_x_avg))
    plt.title('Average')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')

    plt.subplot(153)
    plt.plot(10 * np.log10(MSE_x_loss))
    plt.title('Coordinate-wise median')
    plt.xlabel('iteration')
    plt.ylabel('MSE (dB)')

    plt.show()


    fig2 = plt.figure(figsize=(11, 2.5))
    plt.subplot(151)
    for k in normalAgents:
        plt.plot(W1_no[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylabel(r'$w_{k,i}(1)$', fontsize=25)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(152)
    for k in normalAgents:
        plt.plot(W1_avg[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(153)
    for k in normalAgents:
        plt.plot(W1_loss[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.show()