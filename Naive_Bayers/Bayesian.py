# coding=utf-8
import numpy as np

def createDataSet():
    """
    创建测试的数据集，里面的数值中具有连续值
    :return:
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]

    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']

    # 特征对应的所有可能的情况
    labels_full = {}

    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel

    return dataSet, labels, labels_full


def count_labels(data):
    '''
    :param data:数据集
    :return: 返回好瓜和坏瓜的数目
    '''
    yes = 0
    no = 0
    for s in range(data.__len__()):
        if data[s][-1] == '好瓜':
            yes += 1
        else:
            no += 1
    return yes, no


def handle_one_data(data, attr, location, yes, no, N):
    '''
    :param data: 数据集
    :param attr: 要传入的属性
    :param location: 传入属性的位置
    :param yes: 好瓜数量
    :param no: 坏瓜数量
    :return: 返回该属性在好瓜或者是坏瓜的前提下的概率
    '''
    attr_y, attr_n = 0, 0
    for s in range(data.__len__()):
        if data[s][-1] == '好瓜':
            if data[s][location] == attr:
                attr_y += 1
        else:
            if data[s][location] == attr:
                attr_n += 1
    # 防止某个属性的取值个数为0的概率出现，采用拉普拉斯修正下列概率(各个属性不同取值已经完成如函数count_attr_dis)
    return (attr_y + 1) / (yes + N[location-1]), (attr_n + 1) / (no + N[location-1])


def handle_data(data):
    '''
    :param data: 数据集
    :return: 对密度和含糖率的均值和标准差
    '''
    midu_y = []
    tiandu_y = []
    midu_n = []
    tiandu_n = []
    for s in range(data.__len__()):
        if data[s][-1] == '好瓜':
            midu_y.append(np.float(data[s][-3]))
            tiandu_y.append(np.float(data[s][-2]))
        else:
            midu_n.append(np.float(data[s][-3]))
            tiandu_n.append(np.float(data[s][-2]))
    m_midu_y = np.mean(midu_y)
    m_midu_n = np.mean(midu_n)
    t_tiandu_y = np.mean(tiandu_y)
    t_tiandu_n = np.mean(tiandu_n)
    std_midu_y = np.std(midu_y)
    std_midu_n = np.std(midu_n)
    std_tiandu_y = np.std(tiandu_y)
    std_tiandu_n = np.std(tiandu_n)

    return m_midu_y, m_midu_n, t_tiandu_y, t_tiandu_n, std_midu_y, std_midu_n, std_tiandu_y, std_tiandu_n


def show_result(p_yes, p_no):
    '''
    :param p_yes: 在好瓜的前提下，测试数据各个属性的概率
    :param p_no: 在是坏瓜的前提下，测试数据的各个属性的概率
    :return: 是好瓜或者是坏瓜
    '''
    p1 = 1.0
    p2 = 1.0
    for s in range(p_yes.__len__()):
        p1 *= np.float(p_yes[s])
        p2 *= np.float(p_no[s])
    if p1 > p2:
        print("好瓜", p1, p2)
    else:
        print("坏瓜", p1, p2)


def count_attr_dis(data):
    '''
    :param data: 数据集
    :return: 各个属性取值的个数
    '''
    count = []  # 记录各个属性的取值有多少个不同
    for i in range(data[0].__len__()):
        if i == 0 or i == 7 or i == 8:  # 去掉编号，密度，甜度这个属性
            continue
        d = []
        for s in range(data.__len__()):
            if not d.__contains__(data[s][i]):  # 如果读到的属性不包含在d里就加入到d中
                d.append(data[s][i])
        count.append(d.__len__())  # 统计属性取值不同的个数
    return count


if __name__ == '__main__':
    (data, labels, labels_full) = createDataSet()
    m_midu_y, m_midu_n, t_tiandu_y, t_tiandu_n, std_midu_y, std_midu_n, std_tiandu_y, std_tiandu_n = handle_data(data)
    yes, no = count_labels(data)
    p_yes = [yes / (yes + no)]
    p_no = [no / (yes + no)]
    test_data = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]
    N = []
    N = count_attr_dis(data)
    for s in range(6):
        s_yes, s_no = handle_one_data(data, test_data[s], s + 1, yes, no, N)
        p_yes.append(s_yes)
        p_no.append(s_no)

    p_yes.append(
        1 / (np.sqrt(2 * np.pi) * std_midu_y) * np.exp((-1) * ((test_data[6] - m_midu_y) ** 2) / std_midu_y ** 2))
    p_no.append(
        1 / (np.sqrt(2 * np.pi) * std_midu_n) * np.exp((-1) * ((test_data[6] - m_midu_n) ** 2) / std_midu_n ** 2))

    p_yes.append(
        1 / (np.sqrt(2 * np.pi) * std_tiandu_y) * np.exp((-1) * ((test_data[7] - t_tiandu_y) ** 2) / std_tiandu_y ** 2))
    p_no.append(
        1 / (np.sqrt(2 * np.pi) * std_tiandu_n) * np.exp((-1) * ((test_data[7] - t_tiandu_n) ** 2) / std_tiandu_n ** 2))

    print(p_yes)
    print(p_no)
    show_result(p_yes, p_no)

    # 防止某个属性的取值个数为0的概率出现，采用拉皮拉斯修正(各个属性不同取值已经完成如函数count_attr_dis)

    print(N, '不同属性取值')
