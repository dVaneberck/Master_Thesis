import matplotlib.pyplot as plt
from datetime import datetime


def load_data(path):
    log = open(path, 'r')
    log.readline()

    episodes = []
    step = []
    mean_reward = []
    mean_length = []
    mean_loss = []
    mean_q_value = []
    time = []
    time_zero = 0
    for line in log:
        data = line.split()

        episodes.append(float(data[0]))
        step.append(float(data[1]))
        if len(mean_reward) == 0:
            mean_reward.append(0.0)
        else:
            mean_reward.append(float(data[3]))
        mean_length.append(float(data[4]))
        mean_loss.append(float(data[5]))
        mean_q_value.append(float(data[6]))
        if len(time) == 0:
            time.append(0.0)
            time_zero = data[8]
            time_zero = datetime.strptime(time_zero, '%Y-%m-%dT%H:%M:%S')
        else:
            new_time = datetime.strptime(data[8], '%Y-%m-%dT%H:%M:%S')
            delta_time = new_time - time_zero
            time.append(delta_time.total_seconds() / 60)
            # if delta_time.total_seconds() / 60 > 20:
            #     break
    return episodes, step, mean_reward, mean_length, mean_loss, mean_q_value, time


if __name__ == "__main__":
    data1 = load_data("checkpoints_cartpole_num/2021-08-06T20-16-43-ALL/log")
    data2 = load_data("checkpoints_cartpole_num/2021-08-06T19-19-16-DDQN-EPSI/log")
    # data3 = load_data("checkpoints_cartpole_num/2021-08-07T10-59-17-PER-DDQN/log")
    # data4 = load_data("checkpoints_cartpole_num/2021-08-07T12-30-11-PER/log")

    plt.plot(data1[6], data1[2])
    plt.plot(data2[6], data2[2])
    # plt.plot(data3[6], data3[2])
    # plt.plot(data4[6], data4[2])

    label_1 = "PER"
    label_2 = "Memory Replay"
    # label_3 = "PER + DDQN"
    # label_4 = "PER"
    # plt.legend([label_1])
    plt.legend([label_1, label_2])

    label_x = "Time in minutes"
    label_y = "Rewards"
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    plt.savefig("img/cartpole_per_mem.jpg")
    plt.show()
    plt.clf()
