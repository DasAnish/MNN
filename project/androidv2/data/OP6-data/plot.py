import matplotlib.pyplot as plt
import numpy as np

import os

def get_nord_data():
    files = []
    for i in os.walk(os.getcwd()):
        files = i[2]
        break

    files = [i for i in files if '.txt' in i]
    # files = [i for i in files if i != 'empty.txt']
    files = [i for i in files if "OP6" in i]
    # print(files)
    # for i in files:
    #     print(i)

    data = {} # filename: tuple-2(list(size), list(time))
    for file in files:
        # file =
        size_temp = []
        time_temp = []
        temp = time_temp
        data[file.strip('.txt')] = temp
        print(f"============{file}=============")
        with open(file, "r") as f:
            for line in f.readlines():
                line = line.strip('I/MNNJNI: ')
                # print(line)
                size, time = line.split('-> ')
                time = time.strip('\n')

                size = int(size)
                time = float(time)

                # size_temp.append(size)
                time_temp.append(time)
    #    print(temp_)

    sizes = [i*32 for i in range(1, 17)]
    # print('debug1')

    for file in data:
        t = data[file]
        t = t[:len(sizes)]
        data[file] = t
    # print(data)

    sizes = np.array(sizes)

    sorted_data = list(data.keys())
    sorted_data.sort()

    return data, sorted_data, sizes

def plot_line():

    data, sorted_data, sizes = get_nord_data()

    plt.figure(figsize=(10, 10))
    for name in sorted_data:
        if max(data[name]) > 10000: continue
        if 'cpu' in name:
            plt.plot(sizes, data[name], label=name, linewidth=2)
        else:
            plt.plot(sizes, data[name], label=name)
    # print('debug: done', name)

    plt.xticks(sizes)
# print('debug3')
    plt.title("execution time (Us) vs Size")
    plt.legend()
    plt.show()

def plot_flops(bar = False):

    data, sorted_data, sizes = get_nord_data()

    num_opes = 2 * sizes ** 3

    gflops = []
    gflop_dict = {name: [0 for _ in data[name]] for name in data}
    for name in sorted_data:
        for i, num_op in enumerate(num_opes):
            gflop = num_op / data[name][i] / 1000.0 # *10^6 (becauseof time) and / 10^9 for Giga
            gflop_dict[name][i] = gflop

        gflops.append(max(gflop_dict[name]))



    ypos = np.arange(len(sorted_data))

    if bar:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(ypos, gflops)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sorted_data)

        ax.set_title('max GFLOPs')
    else:
        fig, ax = plt.subplots(figsize=(20, 10))
        for name in sorted_data:
            # if max(data[name]) > 15000: continue
            if 'cpu' in name:
                ax.plot(sizes, gflop_dict[name], label=name, linewidth=3, linestyle='dotted')
            else:
                ax.plot(sizes, gflop_dict[name], label=name)

        # ax.axvline(x=160, linestyle='dashed', c='black')
        # ax.axvline(x=288, linestyle='dashed', c='black')

        ax.set_xticks(sizes)
        ax.set_title("GFLOPS")
        ax.legend()

    plt.show()







if __name__ == '__main__':
    # plot_flops_nord()#bar=True)
    plot_nord()



