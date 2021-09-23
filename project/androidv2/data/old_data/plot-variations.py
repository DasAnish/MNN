import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

func = lambda x, m, c: m*x + c 

op6 = "OPSix"
nord = "Nord"
file_name = lambda x:  f'-matmul-ver{x}-gpu-times.csv'
cpu_name = '-matul-mnn-cpu-times.csv'

big = "ODE_BIG: "
little = "ODE_LITTLE: "
default = "ODE_DEFAULT: "


####################################################################################################

def strip_basic_prefix(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip('I/MNNJNI: ').strip('\n')
            yield line#, current_index



ci = lambda x: 4 * np.std(x) / np.mean(x)

def get_avg_cost_times(prefix, version):
    sizes = []
    avg_times = []
    cis = []

    with open(prefix + file_name(version), 'r') as f:
        for line in f.readlines():
            line = line.strip('I/MNNJNI: ').strip('\n')
            if line.startswith('C,'):
                # read the data
                line_split = line.split(',')[1:]
                size = int(line_split[0])
                times = [float(i) for i in line_split[1:-1]]
                sizes.append(size)
                avg_times.append(np.mean(times))
                cis.append(np.std(times))
                # cis.append(ci(times))
    return sizes, avg_times, cis

def get_times(prefix, version, time_type):
    sizes = []
    times = []

    for line in strip_basic_prefix(prefix + file_name(version)):
        if line.startswith(time_type):
            line_split = [i for i in line.split(',') if i!='']
            size = line_split[1]
            times_ = [float(i) for i in line_split[2:-1]]

            sizes.append(size)
            times.append(np.mean(times_))
    return np.array(sizes), np.array(times)



def get_squared_error(prefix, version, time_type):
    sizes = []
    errors = []

    with open(prefix + file_name(version), 'r') as f:
        for line in f.readlines():
            line = line.strip('I/MNNJNI: ').strip('\n')
            if line.startswith(time_type):
                line_split = line.split(',')[1:]
                size = int(line_split[0])
                times = [float(i) for i in line_split[1:-1]]
                
                sizes.append(size)
                
                x = np.arange(len(times))
                (m, c), _ = curve_fit(func, x, times)
                # slopes.append(m)
                error = np.array([times[i] - func(i, m, c) for i in x])
                error = np.mean(error*error)
                errors.append(error)

    return sizes, errors


def get_cpu_times(prefix, time_type):
    sizes = []
    times = []
    width_prefix = "WIDTH: "

    for line in strip_basic_prefix(prefix + cpu_name):
        if line.startswith(width_prefix):
            current_index = int(line[len(width_prefix):])
            # print(current_index, end=", ")
            sizes.append(current_index)

        elif line.startswith(time_type):
            line_split = line.strip(time_type).split(',')
            line_split = [float(i) for i in line_split if (i != '')]

            
            # print(np.sum(lines_split))
            times.append(np.mean(line_split))
        
    return np.array(sizes), np.array(times)


def plot_all_the_times():
    prefixes = [nord, op6]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    for k, prefix in enumerate(prefixes):
        times = {
            "cpu-big": get_cpu_times(prefix, big),
            "cpu-little": get_cpu_times(prefix, little),
            "cpu-default": get_cpu_times(prefix, default),
            "gpu-ver1": get_times(prefix, 1, 'C'),
            "gpu-ver2": get_times(prefix, 2, 'C')
        }

        sizes = times['cpu-big'][0][1:]

        for label in times:
            times[label] = times[label][1][1:] if "cpu" in label else times[label][1][:-1]

        for label in times:
            # if prefix == nord and label == 'gpu-ver1': continue
            # if prefix == op6 and label == 'cpu-little': continue
            axs[k].plot(sizes, times[label], label=label)
        axs[k].legend()
        axs[k].set_xticks([i for i in range(0, len(sizes)+32, 32)])

    plt.show()

                    

def plot_percentage_cost_time():
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    # sizes, cost_times, _ = get_avg_cost_times(nord, 1)

    prefixes = [nord, op6]
    versions = [1, 2]

    for i1, prefix in enumerate(prefixes):
        for i2, version in enumerate(versions):


            sizes, queued_time = get_times(prefix, version, 'Q')
            _, submit_times = get_times(prefix, version, 'S')
            _, cost_times = get_times(prefix, version, 'C')

            output = cost_times / (submit_times + queued_time + cost_times)


            axs[i1, i2].plot(sizes, output)
            axs[i1, i2].set_title('Nord ver1')
            axs[i1, i2].set_xticks([i for i in range(0, len(sizes), 16)])

    

    plt.show()



def plot_cost_time_analysis():
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    sizes, avg_times, std1 = get_avg_cost_times(nord, 1)
    _, avg_times2, std2 = get_avg_cost_times(nord, 2)

    sizes = np.array(sizes)
    x = (sizes * sizes * 3 * 4)
    a1 = np.array(avg_times)
    a2 = np.array(avg_times2)

    axs[0, 0].plot(x, a1, label='Ver1')
    axs[0, 0].plot(x, a2, label='Ver2')
    axs[0, 0].set_title('Nord mean cost times')
    axs[0, 0].legend()

    axs[0, 1].plot(x, std1/a1, label='std1')
    axs[0, 1].plot(x, std2/a2, label='std2')
    axs[0, 1].set_title('Nord std_cost_time / avg_cost_time')
    axs[0, 1].legend()
    # axs[0].set_xlabel('Total size in bytes of 3 matrixes')
    _, avg_times, std1 = get_avg_cost_times(op6, 1)
    _, avg_times2, std2 = get_avg_cost_times(op6, 2)

    x = (sizes * sizes * 3 * 4)
    a1 = np.array(avg_times)
    a2 = np.array(avg_times2)

    axs[1, 0].plot(x, a1, label='Ver1')
    axs[1, 0].plot(x, a2, label='Ver2')
    axs[1, 0].set_title('OP6 cost times')
    axs[1, 0].legend()

    axs[1, 1].plot(x, std1/a1, label='std1')
    axs[1, 1].plot(x, std2/a2, label='std2')
    axs[1, 1].set_title('OP6 std_cost_time / avg_cost_time')
    axs[1, 1].legend()

    plt.xlabel('Total size in bytes of 3 matrixes')
    plt.ylabel('Cost time in ms')
    # plt.legend()
    plt.show()

def plot_squared_error_variation():
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    # nord 1 submit
    sizes, errors = get_squared_error(nord, 1, 'S')
    axs[0, 0].plot(sizes, errors)
    axs[0, 0].set_title('nord 1 submit')  
    # nord 1 queued
    _, errors = get_squared_error(nord, 1, 'Q')
    axs[0, 1].plot(sizes, errors)
    axs[0, 1].set_title('nord 1 queued') 
    # nord 2 submit
    _, errors = get_squared_error(nord, 2, 'S')
    axs[0, 2].plot(sizes, errors)
    axs[0, 2].set_title('nord 2 submit') 
    # nord 2 queued
    _, errors = get_squared_error(nord, 2, 'Q')
    axs[0, 3].plot(sizes, errors)
    axs[0, 3].set_title('nord 2 queued') 

    # op6 1 sumbmit
    _, errors = get_squared_error(op6, 1, 'S')
    axs[1, 0].plot(sizes, errors)
    axs[1, 0].set_title('OP6 1 submit') 
    # op6 1 queued
    _, errors = get_squared_error(op6, 1, 'Q')
    axs[1, 1].plot(sizes, errors)
    axs[1, 1].set_title('OP6 1 queued') 
    # op6 2 submit
    _, errors = get_squared_error(op6, 2, 'S')
    axs[1, 2].plot(sizes, errors)
    axs[1, 2].set_title('OP6 2 submit') 
    # op6 2 queued
    _, errors = get_squared_error(op6, 2, 'Q')
    axs[1, 3].plot(sizes, errors)
    axs[1, 3].set_title('OP6 2 queued') 

    plt.show()


if __name__ == "__main__":
    # plot_percentage_cost_time()
    # plot_cost_time_analysis()
    # plot_squared_error_variation()
    # get_cpu_times(nord, big)
    plot_all_the_times()            
            


