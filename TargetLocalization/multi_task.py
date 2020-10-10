import copy
import os
from matplotlib import rc
from utils import *

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


def noncooperative_learn(i, numAgents, psi, w, mu_k, q, attackers, psi_a):
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


def average_learn(i, numAgents, psi, w, mu_k, q, attackers, psi_a, Neigh):
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


def loss_learn(i, numAgents, x, u, d, psi, w, mu_k, q, attackers, psi_a, Accumulated_Loss, Neigh):
    a = 0
    gamma = 0.1

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
            loss = (d[k] + np.dot([x[k].x, x[k].y], u[:, k].T).item() - (np.dot([psi[:, k]], u[:, k].T)).item()) ** 2
            Accumulated_Loss[k, k] = (1 - gamma) * Accumulated_Loss[k, k] + gamma * loss
            for l in Neigh[k]:
                if not l == k:
                    loss = (d[k] + np.dot([x[l].x, x[l].y], u[:, k].T).item() - (
                        np.dot([psi[:, l]], u[:, k].T)).item()) ** 2
                    Accumulated_Loss[k, l] = (1 - gamma) * Accumulated_Loss[k, l] + gamma * loss
                if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
                    reversed_loss[l] = (1. / Accumulated_Loss[k, l])
            sum_reversedLoss = sum(reversed_loss)
            for l in Neigh[k]:
                if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
                    weight = reversed_loss[l] / sum_reversedLoss
                    Weight[l] = weight
            # print(k, Weight)
            w[:, k] = np.dot(psi, Weight)
        else:
            w[:, k] = psi[:, k]

    # for k in range(numAgents):
    #     if k not in attackers:
    #         Weight = np.zeros((numAgents,))
    #         reversed_loss = np.zeros((numAgents,))
    #         loss = (w[0, k] - psi[0, k])**2 + (w[1, k] - psi[1, k])** 2
    #         Accumulated_Loss[k, k] = (1 - gamma) * Accumulated_Loss[k, k] + gamma * loss
    #         for l in Neigh[k]:
    #             loss = (w[0, k] - psi[0, l])**2 + (w[1, k] - psi[1, l])** 2
    #             Accumulated_Loss[k, l] = (1 - gamma) * Accumulated_Loss[k, l] + gamma * loss
    #             if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
    #                 reversed_loss[l] = (1. / Accumulated_Loss[k, l])
    #         sum_reversedLoss = sum(reversed_loss)
    #         for l in Neigh[k]:
    #             if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
    #                 weight = reversed_loss[l] / sum_reversedLoss
    #                 Weight[l] = weight
    #         #print(k, Weight)
    #         w[:, k] = np.dot(psi, Weight)
    #     else:
    #         w[:, k] = psi[:, k]

    return w, Accumulated_Loss


def distance_learn(i, numAgents, x, u, d, psi, w, mu_k, q, attackers, psi_a, Accumulated_Loss, Neigh):
    a = 0
    gamma = 0.1

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
            loss = (w[0, k] - psi[0, k]) ** 2 + (w[1, k] - psi[1, k]) ** 2
            Accumulated_Loss[k, k] = (1 - gamma) * Accumulated_Loss[k, k] + gamma * loss
            for l in Neigh[k]:
                if not l == k:
                    loss = (w[0, k] - psi[0, l]) ** 2 + (w[1, k] - psi[1, l]) ** 2
                    Accumulated_Loss[k, l] = (1 - gamma) * Accumulated_Loss[k, l] + gamma * loss
                # if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
                reversed_loss[l] = (1. / Accumulated_Loss[k, l])
            sum_reversedLoss = sum(reversed_loss)
            for l in Neigh[k]:
                # if Accumulated_Loss[k, l] <= Accumulated_Loss[k, k]:
                weight = reversed_loss[l] / sum_reversedLoss
                Weight[l] = weight
            # print(k, Weight)
            w[:, k] = np.dot(psi, Weight)
        else:
            w[:, k] = psi[:, k]

    return w, Accumulated_Loss


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    # parameters
    iteration = 500

    r = 1
    box = 12
    numAgents = 100
    mu_k = 0.1

    t = 1
    w0 = [Point(10 + t * random.random(), 10 + t * random.random()),
          Point(10 + t * random.random(), 20 + t * random.random()),
          Point(20 + t * random.random(), 10 + t * random.random()),
          Point(20 + t * random.random(), 20 + t * random.random())]
    print([(p.x, p.y) for p in w0])
    W0 = [w0[0]] * (numAgents // 4) + [w0[1]] * (numAgents // 4) + [w0[2]] * (numAgents // 4) + [w0[3]] * (
            numAgents // 4)

    lower = 0
    upper = 3
    sensingRange = 1

    x_no = random_point_set(numAgents, lower=lower, upper=upper)
    # for k in attackers:
    #     w_no[k] = Point(np.random.random(), np.random.random())

    x_init = copy.deepcopy(x_no)
    x_avg = copy.deepcopy(x_no)
    x_loss = copy.deepcopy(x_no)
    x_dist = copy.deepcopy(x_no)

    attackerNum = 20

    attackers = random.sample(list(range(numAgents)), attackerNum)

    normalAgents = [k for k in range(numAgents) if k not in attackers]

    Neigh = []
    for k in range(numAgents):
        neighbor = findNeighbors(x_init, k, numAgents, sensingRange, maxNeighborSize=10)
        Neigh.append(neighbor)

    fig = plt.figure(figsize=(4, 3))
    # plt.grid(True, which='major')
    ax = plt.gca()
    ax.set_xlim(lower - 0.1, upper + 0.1)
    ax.set_ylim(lower - 0.1, upper + 0.1)
    # plt.xticks([0.3*i for i in range(-5,5, 1)])
    # plt.yticks([0.3*i for i in range(-5,5, 1)])
    plt.gca().set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='dotted')
    # for tic in ax.xaxis.get_major_ticks():
    #     tic.tick1On = tic.tick2On = False
    #     tic.label1On = tic.label2On = False
    # for tic in ax.yaxis.get_major_ticks():
    #     tic.tick1On = tic.tick2On = False
    #     tic.label1On = tic.label2On = False
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end + 0.4, 0.5))
    ax.yaxis.set_ticks(np.arange(0, end + 0.4, 0.5))

    # ax.set_xticks([0, 0.3, 0.4, 1.0, 1.5])
    # ax.set_xticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    # ax.set_yticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    for i in range(0, numAgents):
        for neighbor in Neigh[i]:
            plt.plot([x_init[i].x, x_init[neighbor].x], [x_init[i].y, x_init[neighbor].y], linewidth=0.1,
                     color='gray')

    plot_point_set(x_init[:numAgents // 4], color='b')  # fault-free robots are plotted in blue
    plot_point_set(x_init[numAgents // 4:numAgents // 2], color='g')  # fault-free robots are plotted in blue
    plot_point_set(x_init[numAgents // 2:numAgents // 4 * 3], color='m')  # fault-free robots are plotted in blue
    plot_point_set(x_init[numAgents // 4 * 3:], color='y')  # fault-free robots are plotted in blue
    plot_point_set([x_init[p] for p in attackers], color='r')  # faulty robots are plotted in red
    plt.savefig('fig/network_attackerNum%d.png' % attackerNum)

    # plt.pause(0.1)
    plt.title('Network Connectivity')
    plt.show()
    # plt.savefig('./result/largeNetwork/%s%d.eps' % (method, t))
    # end = input('Press enter to end the program.')

    fig4 = plt.figure(figsize=(4, 3))
    # plt.grid(True, which='major')
    ax = plt.gca()
    # ax.set_xlim(lower-0.1, upper+0.1)
    # ax.set_ylim(lower-0.1, upper+0.1)
    # plt.xticks([0.3*i for i in range(-5,5, 1)])
    # plt.yticks([0.3*i for i in range(-5,5, 1)])
    # plt.gca().set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='dotted')
    # for tic in ax.xaxis.get_major_ticks():
    #     tic.tick1On = tic.tick2On = False
    #     tic.label1On = tic.label2On = False
    # for tic in ax.yaxis.get_major_ticks():
    #     tic.tick1On = tic.tick2On = False
    #     tic.label1On = tic.label2On = False
    # start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(start, end + 0.01, 0.2))
    # ax.yaxis.set_ticks(np.arange(start, end + 0.01, 0.2))

    # ax.set_xticks([0, 0.3, 0.4, 1.0, 1.5])
    # ax.set_xticklabels([-1, "", "", "", "", 0, "", "", "", "", 1])
    # ax.set_yticklabels([-1, "", "", "", "", 0, "", "", "", "", 1])

    plot_point(w0[0], color='b')  # fault-free robots are plotted in blue
    plot_point(w0[1], color='g')  # fault-free robots are plotted in blue
    plot_point(w0[2], color='m')  # fault-free robots are plotted in blue
    plot_point(w0[3], color='y')  # fault-free robots are plotted in blue
    plt.title('Targets Position')
    plt.show()

    # psi_a = 0 * np.ones((2, len(attackers)))
    psi_a = 15 + np.random.random((2, len(attackers)))

    mu_vd = 0
    mu_vu = 0
    # sigma_vd2 = 0 + 0 * np.random.random((numAgents, 1))
    # # sigma_vd2[random.sample(range(numAgents), 20)] = 5
    # sigma_vu2 = 0 + 0 * np.random.random((numAgents, 1))
    # # sigma_vu2[random.sample(range(numAgents), 5)] = 0.3
    sigma_vd2 = 0.1 + 0.1 * np.random.random((numAgents, 1))
    # sigma_vd2[random.sample(range(numAgents), 20)] = 5
    sigma_vu2 = 0.01 + 0.01 * np.random.random((numAgents, 1))
    sigma_vu2[random.sample(range(numAgents), 5)] = 0.1

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
        w_no[0, k], w_no[1, k] = np.random.random(), np.random.random()
    w_avg = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_avg[0, k], w_avg[1, k] = w_no[0, k], w_no[1, k]
    w_loss = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_loss[0, k], w_loss[1, k] = w_no[0, k], w_no[1, k]
    w_dist = np.zeros((2, numAgents))
    for k in range(numAgents):
        w_dist[0, k], w_dist[1, k] = w_no[0, k], w_no[1, k]

    Loss_no = np.zeros((iteration, numAgents))
    Loss_avg = np.zeros((iteration, numAgents))
    Loss_loss = np.zeros((iteration, numAgents))
    Loss_dist = np.zeros((iteration, numAgents))

    W1_no = np.zeros((iteration, numAgents))
    W1_avg = np.zeros((iteration, numAgents))
    W1_loss = np.zeros((iteration, numAgents))
    W1_dist = np.zeros((iteration, numAgents))

    Accumulated_Loss = 10 * np.ones((numAgents, numAgents))
    Accumulated_dist = 10 * np.ones((numAgents, numAgents))

    for i in range(iteration):

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

        for k in range(numAgents):
            if k in attackers:
                continue
            dist = W0[k].distance(x_init[k])
            unit = [(W0[k].x - x_init[k].x) / dist, (W0[k].y - x_init[k].y) / dist]
            u[:, k] = unit + vu[i, k]
            d[k] = np.dot([W0[k].x - x_init[k].x, W0[k].y - x_init[k].y], u[:, k].T) + vd[i, k]
            q[:, k] = [x_init[k].x, x_init[k].y] + d[k] * u[:, k]

        # noncooperative
        w_no = noncooperative_learn(i, numAgents, psi, w_no, mu_k, q, attackers, psi_a)

        # cooperative
        w_avg = average_learn(i, numAgents, psi, w_avg, mu_k, q, attackers, psi_a, Neigh)

        w_loss, Accumulated_Loss = loss_learn(i, numAgents, x_loss, u, d, psi, w_loss, mu_k, q, attackers, psi_a,
                                              Accumulated_Loss, Neigh)

        w_dist, Accumulated_dist = distance_learn(i, numAgents, x_dist, u, d, psi, w_dist, mu_k, q, attackers, psi_a,
                                                  Accumulated_dist, Neigh)

        # loss_no = 0
        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_no[0, k], w_no[1, k])
            # error_no += agent.distance(W0[k]) ** 2
            loss_no = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_no[:, k]], u[:, k].T)).item()) ** 2
            W1_no[i, k] = w_no[0, k]
            Loss_no[i, k] = loss_no

        # loss_avg = 0
        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_avg[0, k], w_avg[1, k])
            # error_avg += agent.distance(W0[k]) ** 2
            loss_avg = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_avg[:, k]], u[:, k].T)).item()) ** 2
            W1_avg[i, k] = w_avg[0, k]
            Loss_avg[i, k] = loss_avg

        # loss_loss = 0
        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_loss[0, k], w_loss[1, k])
            # error_loss += (agent.distance(W0[k])) ** 2
            loss_loss = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_loss[:, k]], u[:, k].T)).item()) ** 2
            W1_loss[i, k] = w_loss[0, k]
            Loss_loss[i, k] = loss_loss

        for k in range(numAgents):
            if k in attackers:
                continue
            agent = Point(w_dist[0, k], w_dist[1, k])
            # error_loss += (agent.distance(W0[k])) ** 2
            loss_dist = (d[k] + np.dot([x_init[k].x, x_init[k].y], u[:, k].T).item() - (
                np.dot([w_dist[:, k]], u[:, k].T)).item()) ** 2
            W1_dist[i, k] = w_dist[0, k]
            Loss_dist[i, k] = loss_dist

        print('iteration %d' % i)

    print("Loss_no_mean =", np.mean(np.delete(Loss_no, attackers, axis=1), 1))
    # print("Loss_no_var", np.sqrt(np.var(np.delete(Loss_no, attackers, axis=1), 1)))
    print("Loss_no_min =", np.min(np.delete(Loss_no, attackers, axis=1), 1))
    print("Loss_no_max =", np.max(np.delete(Loss_no, attackers, axis=1), 1))

    print("Loss_avg_mean =", np.mean(np.delete(Loss_avg, attackers, axis=1), 1))
    # print("Loss_avg_var", np.sqrt(np.var(np.delete(Loss_avg, attackers, axis=1), 1)))
    print("Loss_avg_min =", np.min(np.delete(Loss_avg, attackers, axis=1), 1))
    print("Loss_avg_max =", np.max(np.delete(Loss_avg, attackers, axis=1), 1))

    print("Loss_loss_mean =", np.mean(np.delete(Loss_loss, attackers, axis=1), 1))
    # print("Loss_loss_var", np.sqrt(np.var(np.delete(Loss_loss, attackers, axis=1), 1)))
    print("Loss_loss_min =", np.min(np.delete(Loss_loss, attackers, axis=1), 1))
    print("Loss_loss_max =", np.max(np.delete(Loss_loss, attackers, axis=1), 1))

    print("Loss_dist_mean =", np.mean(np.delete(Loss_dist, attackers, axis=1), 1))
    # print("Loss_dist_var", np.sqrt(np.var(np.delete(Loss_dist, attackers, axis=1), 1)))
    print("Loss_dist_min =", np.min(np.delete(Loss_dist, attackers, axis=1), 1))
    print("Loss_dist_max =", np.max(np.delete(Loss_dist, attackers, axis=1), 1))

    if len(attackers) == 0:
        np.save('results/Loss_loss.npy', Loss_loss)
        np.save('results/Loss_avg.npy', Loss_avg)
        np.save('results/Loss_no.npy', Loss_no)
        np.save('results/Loss_dist.npy', Loss_dist)


    else:
        try:
            os.makedirs("results/attacked_num_%d" % len(attackers))
        except OSError:
            print("Creation of the directory %s failed")
        np.save('results/attacked_num_%d/Loss_loss.npy' % len(attackers), Loss_loss)
        np.save('results/attacked_num_%d/Loss_avg.npy' % len(attackers), Loss_avg)
        np.save('results/attacked_num_%d/Loss_no.npy' % len(attackers), Loss_no)

    fig1 = plt.figure(figsize=(3.9, 2.5))
    # fig1 = plt.figure(figsize=(3.9, 2))
    plt.plot(np.log10(np.mean(np.delete(Loss_no, attackers, axis=1), 1)), label=r'Non-coop')
    plt.plot(np.log10(np.mean(np.delete(Loss_avg, attackers, axis=1), 1)), label=r'Average')
    plt.plot(np.log10(np.mean(np.delete(Loss_loss, attackers, axis=1), 1)), label=r'loss-based')
    plt.plot(np.log10(np.mean(np.delete(Loss_dist, attackers, axis=1), 1)), label=r'dist-based')

    # plt.plot(MSE_x_no[1:], label=r'Non-coop')
    # plt.plot((MSE_x_avg[1:]), label=r'Average')
    # plt.plot((MSE_x_loss[1:]), label=r'loss-based')

    # plt.title('cooperative under attack using median')
    plt.xlabel(r'iteration $i$', fontsize=10)
    plt.ylabel(r'Loss', fontsize=10)
    # plt.xticks([0, 100, 200, 300, 400, 500])
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
    # fig1.savefig('fig/MSD_mobile_attack%d.eps' % attackerNum)

    fig2 = plt.figure(figsize=(11, 2.5))
    plt.subplot(151)
    for k in normalAgents:
        plt.plot(np.log10(Loss_no[:, k]))
    plt.xlabel('iteration $i$', fontsize=20)
    plt.ylabel(r'Average Loss', fontsize=25)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(152)
    for k in normalAgents:
        plt.plot(np.log10(Loss_avg[:, k]))
    plt.xlabel('iteration $i$', fontsize=20)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(153)
    for k in normalAgents:
        plt.plot(np.log10(Loss_loss[:, k]))
    plt.xlabel('iteration $i$', fontsize=20)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.subplot(154)
    for k in normalAgents:
        plt.plot(np.log10(Loss_dist[:, k]))
    plt.xlabel('iteration $i$', fontsize=20)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.show()

    fig3 = plt.figure(figsize=(11, 2.5))
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

    plt.subplot(154)
    for k in normalAgents:
        plt.plot(W1_dist[1:, k])
    plt.xlabel('iteration $i$', fontsize=20)
    plt.xticks([0, 100, 200, 300, 400, 500])

    plt.show()
