import os
os.chdir(r'D:\Desktop\UROP2021\MNN\project\androidv2\data\FC-data')

dataLoad = 'dataLoad'
Forward = forward = 'Forward'
LossCalc = lossCalc = 'LossCalc'
Backward = backward = 'Backward'
cpu = 'cpu'
gpu = 'gpu'


def getData(filename='baseline.txt'):
    ret_cpu = {dataLoad: 0, forward: 0, lossCalc: 0, backward: 0}
    ret_gpu = {dataLoad: 0, forward: 0, lossCalc: 0, backward: 0}

    with open(filename) as f:
        data = f.read()

    count_cpu = 0
    count_gpu = 0
    doing_cpu = True
    for line in data.split('\n'):
        if line == "DONE":
            doing_cpu = False
        # print(line)
        line = line.split('|')[2:]
        if not line: continue

        for l in line:
            l.strip()

        output = eval('{' + ','.join(line) + '}')
        # print(output)

        if doing_cpu:
            for i in output:
                ret_cpu[i] += output[i]
            count_cpu += 1
        else:
            for i in output:
                ret_gpu[i] += output[i]
            count_gpu += 1

    ret_cpu = {i:ret_cpu[i]/count_cpu for i in ret_cpu}
    ret_gpu = {i:ret_gpu[i]/count_gpu for i in ret_gpu}
    ret = {'cpu': ret_cpu, 'gpu': ret_gpu}

    return ret


def get_all_data():
    for i in os.walk(os.getcwd()):
        break

    filenames = i[2]
    filenames = [i for i in filenames if not ('py' in i or 'empty' in i)]
    print(filenames)

    output = {}
    for filename in filenames:
        output[filename] = getData(filename)

    return output


def plot_table(output):
    print('''\\begin{table}
    \\centering
    \\begin{tabular}{cccc}
    ''')
    print("filename-type & data-loading-time & forward-time & backward-time \\\\")
    for file in output:
        for type in [cpu, gpu]:
            temp = file.split('.')[0]
            filename = f"{temp}-{type}"


            temp_line = [
                filename,
                '%.3f' % output[file][type][dataLoad],
                '%.3f' % output[file][type][forward],
                '%.3f' % output[file][type][backward]
            ]

            if type == gpu:
                temp_line = [
                    filename,
                    '%.3f(%.1f)' % (output[file][type][dataLoad], output[file][gpu][dataLoad] / output[file][cpu][dataLoad]),
                    '%.3f(%.1f)' % (output[file][type][forward], output[file][gpu][forward] / output[file][cpu][forward]),
                    '%.3f(%.1f)' % (output[file][type][backward], output[file][gpu][backward] / output[file][cpu][backward])
                ]

            print(' & '.join(temp_line), end='\\\\')
            if type == gpu:
                print(' \\midrule')
            else:
                print()

    print('''\\end{tabular}\\end{table}''')


if __name__ == '__main__':
    # print(getData())
    output = get_all_data()
    # for dic in output:
    #     print(dic, " "*(30-len(dic)), ":", output[dic]['cpu'], '\n', ' '*32, output[dic]['gpu'])
    plot_table(output)