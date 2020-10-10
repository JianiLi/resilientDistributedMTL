import numpy as np

# rule = "no-cooperation"
# #rule = "average"
# #rule = "loss"
# #rule = "distance"

# train_test = "train"
train_test = "test"
loss_or_acc = "acc"
attack = True
attackerNum = 10

if attackerNum == 10:
    attacker = [27,12,24,13,1,8,16,15,28,9]
elif attackerNum == 29:
    normal = [4]
    attacker = np.delete(range(0,30),normal)



print("attack: %s, train or test: %s, loss or acc: %s" % (attack, train_test, loss_or_acc))


def print_num(l, string, num=5):
    print(string + "=[", end=' ')
    for i in range(len(l)):
        print("%.7f" % l[i], end='  ')
        if (i + 1) % num == 0:
            print()
    print("]")


for rule in ["no-cooperation", "average", "distance", "loss"]:
    if attack:
        loss = np.load("results/attacked/%d/individual_average_%s_loss_%s.npy" % (attackerNum, train_test, rule))
        acc = np.load("results/attacked/%d/individual_average_%s_acc_%s.npy" % (attackerNum, train_test, rule))
    else:
        loss = np.load("results/individual_average_%s_loss_%s.npy" % (train_test, rule))
        acc = np.load("results/individual_average_%s_acc_%s.npy" % (train_test, rule))

    # loss
    if attack:
        loss = np.delete(loss, attacker, axis=1)
        acc = np.delete(acc, attacker, axis=1)

    mean_loss = np.nanmean(loss, 1)
    max_loss = np.nanmax(loss, 1)
    min_loss = np.nanmin(loss, 1)
    variance_loss = np.nanvar(loss, 1)

    # accuracy
    mean_acc = np.nanmean(acc, 1)
    max_acc = np.nanmax(acc, 1)
    min_acc = np.nanmin(acc, 1)
    variance_acc = np.nanvar(acc, 1)

    # print("mean_loss_%s" % rule, locals()["mean_" + rule])
    # # print("variance_loss_%s" % task, locals()["variance_" + task])
    # print("min_loss_%s" % rule, locals()["min_" + rule])
    # print("max_loss_%s" % rule, locals()["max_" + rule])

    if rule == "no-cooperation":
        rule = "no_cooperation"

    if loss_or_acc == "loss":
        print_num(mean_loss, "mean_loss_%s" % rule, num=5)
        print_num(min_loss, "min_loss_%s" % rule, num=5)
        print_num(max_loss, "max_loss_%s" % rule, num=5)

    # print("mean_acc_%s" % rule, locals()["mean_" + rule + "_acc"])
    # # print("variance_acc_%s" % task, locals()["variance_" + task + "_acc"])
    # print("min_acc_%s" % rule, locals()["min_" + rule + "_acc"])
    # print("max_acc_%s" % rule, locals()["max_" + rule + "_acc"])
    else:
        print_num(mean_acc, "mean_acc_%s" % rule, num=5)
        print_num(min_acc, "min_acc_%s" % rule, num=5)
        print_num(max_acc, "max_acc_%s" % rule, num=5)
