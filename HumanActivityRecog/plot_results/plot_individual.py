import numpy as np
import matplotlib.pyplot as plt
import itertools


def print_num(l, string, num=5):
    print(string+"=[", end=' ')
    for i in range(len(l)):
        print(l[i], end='  ')
        if (i+1) % num == 0:
            print()
    print("]")

# no_coop = np.load("attacked_average_accuracy_no-cooperation.npy")
# no_coop.tolist()
# print(no_coop)
#
# ave = np.load("attacked_average_accuracy_average.npy")
# ave.tolist()
# print(ave)
#
# loss = np.load("attacked_average_accuracy_loss.npy")
# loss.tolist()
# print(loss)
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(no_coop, label="No cooperation")
# ax.plot(ave, label="Average weights")
# ax.plot(loss, label="Loss based weights")
# plt.xlabel("Iteration", fontsize=15)
# plt.ylabel("Average prediction accuracy", fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
# plt.savefig("prediction accuracy_comparison.eps")
#


no_coop = np.load("results/average_train_loss_no-cooperation.npy")
no_coop_acc = np.load("results/average_train_acc_no-cooperation.npy")
#no_coop = 20*np.log(np.load("results/average_train_loss_no-cooperation.npy"))
no_coop.tolist()
#print(no_coop)
# print(no_coop_acc)
print_num(no_coop, "no_coop", num=5)

ave = np.load("results/average_train_loss_average.npy")
ave_acc = np.load("results/average_train_acc_average.npy")
#ave = 20*np.log(np.load("results/average_train_loss_average.npy"))

ave.tolist()
# print(ave)
print(ave_acc)
loss = np.load("results/average_train_loss_loss.npy")
loss_acc = np.load("results/average_train_acc_loss.npy")

#loss = 20*np.log(np.load("results/average_train_loss_loss.npy"))
loss.tolist()
# print(loss)
print(loss_acc)

distance = np.load("results/average_train_loss_distance.npy")
distance_acc = np.load("results/average_train_acc_distance.npy")
#loss = 20*np.log(np.load("results/average_train_loss_loss.npy"))
distance.tolist()
# print(loss)
print(distance_acc)


# reverse_loss = np.load("results/average_train_loss_reversedLoss.npy")
# reverse_loss_acc = np.load("results/average_train_acc_reversedLoss.npy")
# #loss =  20*np.log(np.load("results/average_train_loss_loss.npy"))
# reverse_loss.tolist()
# print(reverse_loss)

no_coop_test = np.load("results/average_test_loss_no-cooperation.npy")
no_coop_acc_test = np.load("results/average_test_acc_no-cooperation.npy")

#no_coop_test = 20*np.log(np.load("results/average_test_loss_no-cooperation.npy"))
no_coop_test.tolist()
# print(no_coop_test)
print(no_coop_acc_test)

ave_test = np.load("results/average_test_loss_average.npy")
ave_acc_test = np.load("results/average_test_acc_average.npy")

ave_test.tolist()
# print(ave_test)
print(ave_acc_test)
loss_test = np.load("results/average_test_loss_loss.npy")
loss_acc_test = np.load("results/average_test_acc_loss.npy")

#loss_test =  20*np.log(np.load("results/average_test_loss_loss.npy"))
loss_test.tolist()
# print(loss_test)
print(loss_acc_test)

distance_test = np.load("results/average_test_loss_distance.npy")
distance_acc_test = np.load("results/average_test_acc_distance.npy")

#loss_test =  20*np.log(np.load("results/average_test_loss_loss.npy"))
distance_test.tolist()
# print(loss_test)
print(distance_acc_test)


# reverse_loss_test = np.load("results/average_test_loss_reversedLoss.npy")
# reverse_loss_acc_test = np.load("results/average_test_acc_reversedLoss.npy")
# #loss_test =  20*np.log(np.load("results/average_test_loss_loss.npy"))
# reverse_loss_test.tolist()
# print(reverse_loss_test)

fig = plt.figure(figsize=(10,2.5))
ax = fig.add_subplot(1, 4, 1)
ax.plot(no_coop, label="No cooperation")
ax.plot(ave, label="Average weights")
ax.plot(loss, label="Loss based weights")
ax.plot(distance, label="Distance based weights")
#ax.plot(reverse_loss, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average training loss", fontsize=15)
plt.legend(fontsize=15)

ax = fig.add_subplot(1, 4, 2)
ax.plot(no_coop_acc, label="No cooperation")
ax.plot(ave_acc, label="Average weights")
ax.plot(loss_acc, label="Loss based weights")
ax.plot(distance_acc, label="Distance based weights")
#ax.plot(reverse_loss_acc, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average training accuracy", fontsize=15)
# plt.legend(fontsize=15)

ax = fig.add_subplot(1, 4, 3)
ax.plot(no_coop_test, label="No cooperation")
ax.plot(ave_test, label="Average weights")
ax.plot(loss_test, label="Loss based weights")
ax.plot(distance_test, label="Distance based weights")
#ax.plot(reverse_loss_test, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average testing loss", fontsize=15)
# plt.legend(fontsize=15)

ax = fig.add_subplot(1, 4, 4)
ax.plot(no_coop_acc_test, label="No cooperation")
ax.plot(ave_acc_test, label="Average weights")
ax.plot(loss_acc_test, label="Loss based weights")
ax.plot(distance_acc_test, label="Distance based weights")
#ax.plot(reverse_loss_acc_test, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average testing accuracy", fontsize=15)
# plt.legend(fontsize=15)

plt.show()
plt.savefig("prediction accuracy_comparison.eps")



# --------------------- Under attack below -----------------------

no_coop = np.load("results/attacked/average_train_loss_no-cooperation.npy")
no_coop_acc = np.load("results/attacked/average_train_acc_no-cooperation.npy")

#no_coop = 20*np.log(np.load("results/attacked/average_train_loss_no-cooperation.npy"))
no_coop.tolist()
print("loss-no_coop", no_coop)
print("acc-no_coop", no_coop_acc)


ave = np.load("results/attacked/average_train_loss_average.npy")
ave_acc = np.load("results/attacked/average_train_acc_average.npy")

ave.tolist()
print("loss-ave", ave)
print("acc-ave", ave_acc)

loss = np.load("results/attacked/average_train_loss_loss.npy")
loss_acc = np.load("results/attacked/average_train_acc_loss.npy")
#loss =  20*np.log(np.load("results/attacked/average_train_loss_loss.npy"))
loss.tolist()
print("loss-loss", loss)
print("acc-loss", loss_acc)

distance = np.load("results/attacked/average_train_loss_distance.npy")
distance_acc = np.load("results/attacked/average_train_acc_distance.npy")

#loss =  20*np.log(np.load("results/attacked/average_train_loss_loss.npy"))
distance.tolist()
print("loss-loss", distance)
print("acc-loss", distance_acc)



# reverse_loss = np.load("results/attacked/average_train_loss_reversedLoss.npy")
# reverse_loss_acc = np.load("results/attacked/average_train_acc_reversedLoss.npy")
# #loss =  20*np.log(np.load("results/attacked/average_train_loss_loss.npy"))
# reverse_loss.tolist()
# print(reverse_loss)

no_coop_test = np.load("results/attacked/average_test_loss_no-cooperation.npy")
no_coop_acc_test = np.load("results/attacked/average_test_acc_no-cooperation.npy")

#no_coop_test = 20*np.log(np.load("results/attacked/average_test_loss_no-cooperation.npy"))
no_coop_test.tolist()
print("loss-no_coop_test", no_coop_test)
print("acc-no_coop_test", no_coop_acc_test)


ave_test = np.load("results/attacked/average_test_loss_average.npy")
ave_acc_test = np.load("results/attacked/average_test_acc_average.npy")

ave_test.tolist()
print("loss-ave_test",ave_test)
print("acc-ave_test",ave_acc_test)

loss_test = np.load("results/attacked/average_test_loss_loss.npy")
loss_acc_test = np.load("results/attacked/average_test_acc_loss.npy")
#loss_test =  20*np.log(np.load("results/attacked/average_test_loss_loss.npy"))
loss_test.tolist()
print("loss-loss_test",loss_test)
print("acc-loss_test",loss_acc_test)


distance_test = np.load("results/attacked/average_test_loss_distance.npy")
distance_acc_test = np.load("results/attacked/average_test_acc_distance.npy")
#loss_test =  20*np.log(np.load("results/attacked/average_test_loss_loss.npy"))
distance_test.tolist()
print("distance-loss_test",distance_test)
print("acc-distance_test",distance_acc_test)


fig = plt.figure(figsize=(10,2.5))
ax = fig.add_subplot(1, 4, 1)
ax.plot(no_coop, label="No cooperation")
ax.plot(ave, label="Average weights")
ax.plot(loss, label="Loss based weights")
ax.plot(distance, label="Distance based weights")
#ax.plot(reverse_loss, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average training loss", fontsize=15)
plt.legend(fontsize=15)

ax = fig.add_subplot(1, 4, 2)
ax.plot(no_coop_acc, label="No cooperation")
ax.plot(ave_acc, label="Average weights")
ax.plot(loss_acc, label="Loss based weights")
ax.plot(distance_acc, label="Distance based weights")
#ax.plot(reverse_loss_acc, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average training accuracy", fontsize=15)
# plt.legend(fontsize=15)

ax = fig.add_subplot(1, 4, 3)
ax.plot(no_coop_test, label="No cooperation")
ax.plot(ave_test, label="Average weights")
ax.plot(loss_test, label="Loss based weights")
ax.plot(distance_test, label="Distance based weights")
#ax.plot(reverse_loss_test, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average testing loss", fontsize=15)
# plt.legend(fontsize=15)

ax = fig.add_subplot(1, 4, 4)
ax.plot(no_coop_acc_test, label="No cooperation")
ax.plot(ave_acc_test, label="Average weights")
ax.plot(loss_acc_test, label="Loss based weights")
ax.plot(distance_acc_test, label="Distance based weights")
#ax.plot(reverse_loss_acc_test, label="Reversed Loss based weights")
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Average testing accuracy", fontsize=15)
# plt.legend(fontsize=15)

plt.show()
plt.savefig("attacked_prediction accuracy_comparison.eps")
