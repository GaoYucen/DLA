#%%
import pandas as pd
import numpy as np
from config import get_config
import os
import time

params, _ = get_config()
look_back = params.look_back
look_forward = params.look_forward

#%%
# 读数据集
def readData(filepath):
    data = pd.read_excel(filepath)
    data['时间'] = data['时间'].astype(str)
    df = data.groupby(data['时间'].str[:10])['测量的桥面系挠度值'].mean().reset_index(name='mean_value')
    values = df['mean_value'].values.astype(float)
    return values

#%%
filepath = 'data/Hongfu/deflection/'
# 读取filepath中的文件名
filelist = os.listdir(filepath)
filelist.sort()

#%%

for file in filelist:
    print('node: ', file)
    values = readData(filepath+file)

    #%% 按照8:2划分训练集和测试集
    train_size = int(len(values) * 0.8)
    train, test = values[0:train_size], values[train_size:len(values)]

    #%% 构造train_x, train_y, test_x和test_y，使用24步预测6步
    def create_dataset(dataset, look_back=1, look_forward=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-look_forward+1):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back:i+look_back+look_forward])
        return np.array(dataX), np.array(dataY)

    train_x, train_y = create_dataset(train, look_back, look_forward)
    test_x, test_y = create_dataset(test, look_back, look_forward)

    #%% 使用RF模型进行预测
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    # start = time.time()
    # # Fitting the RF model on the training data
    # model = RandomForestRegressor(bootstrap=True, max_depth=50, max_features=3,
    #                             min_samples_leaf=3, min_samples_split=8,
    #                             n_estimators=100, n_jobs=-1, verbose=2)
    # model.fit(train_x, train_y)
    # end = time.time()
    # print('training time: ', end-start)
    #
    # #%% 存储模型参数
    # import pickle
    # with open('param/RF_'+file+'.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    #%% 读取模型参数
    model = RandomForestRegressor(bootstrap=True, max_depth=50, max_features=3,
                                min_samples_leaf=3, min_samples_split=8,
                                n_estimators=100, n_jobs=-1, verbose=2)

    import pickle
    with open('param/RF_'+file+'.pkl', 'rb') as f:
        model = pickle.load(f)

    #%% 测试
    # Making predictions on the test data
    start = time.time()
    predictions = model.predict(test_x)
    end = time.time()
    print('testing time: ', end-start)

    #%%
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt
    # 计算MAE, MSE, RMSE和R2
    rmse = sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    mse = mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)

    print('Test MAE: %.3f' % mae)
    print('Test RMSE: %.3f' % rmse)
    print('Test MSE: %.3f' % mse)
    print('Test R2: %.3f' % r2)

    #%% 构造和test数据相同长度的prediction数据
    # 假设 predictions 是一个二维 numpy 数组
    predictions = np.array(predictions)

    test_prediction = []
    for i in range(len(test) - look_back):
        if i < look_forward - 1:
            num = i + 1
            indices = [(i - j, j) for j in range(num)]
        elif i > len(test) - look_back - look_forward:
            num = len(test) - look_back - i
            indices = [(i - look_forward + 1 + j, look_forward - 1 - j) for j in range(num)]
        else:
            num = look_forward
            indices = [(i - j, j) for j in range(num)]

        # 使用 numpy 数组索引来获取元素并计算平均值
        values = [predictions[row, col] for row, col in indices]
        avg = np.mean(values)
        test_prediction.append(avg)

    #%% 画图
    fontsize_tmp = 50
    import matplotlib.pyplot as plt

    import matplotlib.dates as mdates

    # 创建日期范围
    dates = pd.date_range(start='2023-06-08', end='2023-12-15')

    # 创建图形
    fig, ax = plt.subplots(figsize=(40, 10))

    # 绘制原始数据和预测数据
    ax.plot(dates, test[look_back:], label='Origin', linestyle='-', linewidth=2)
    ax.plot(dates, test_prediction, label='Prediction', linestyle='-.', linewidth=4)

    # 设置日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xlabel('Date', fontsize=fontsize_tmp)

    # 设置y轴标签
    ax.set_ylabel('Deflection (m)', fontsize=fontsize_tmp)

    # 设置x轴和y轴的刻度字体大小
    plt.xticks(fontsize=fontsize_tmp)
    plt.yticks(fontsize=fontsize_tmp)

    # 显示图例
    plt.legend(prop={'size': fontsize_tmp}, loc='lower left')
    plt.savefig('graph/RF'+file[0:8] + '.pdf', bbox_inches='tight')


