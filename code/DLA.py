#%%
# Importing necessary library
import pandas as pd
import numpy as np
import sys
sys.path.append('model/')
from config import get_config
import os
from PyEMD import EMD
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

params, _ = get_config()
look_back = params.look_back
look_forward = params.look_forward

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

    def create(dataset, split=0.8):
        train_size = int(len(dataset) * split)
        train, test = dataset[0:train_size], dataset[train_size:len(values)]
        train_x, train_y = create_dataset(train, look_back, look_forward)
        test_x, test_y = create_dataset(test, look_back, look_forward)
        return train_x, train_y, test_x, test_y

    #%% 构造LSTM模型进行预测
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from keras.callbacks import EarlyStopping

    def create_LSTM(train_x, train_y, test_x, test_y, model_name='LSTM', Epochs=100):
        # Reshaping the train_x and test_x data
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        # Creating the LSTM model
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(look_forward))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Fitting the LSTM model on the training data
        early_stopping = EarlyStopping(monitor='loss', mode='min', patience=80, verbose=1)
        model.fit(train_x, train_y, epochs=Epochs, validation_data=(test_x, test_y), batch_size=32, verbose=1,
                  callbacks=[early_stopping])

        # Making predictions on the test data
        predictions = model.predict(test_x)

        # Calculating the root mean squared error
        rmse = sqrt(mean_squared_error(test_y, predictions))
        print('Test RMSE: %.3f' % rmse)

        model.save('param/DLA'+file+model_name+'.h5')

        return predictions

    #%% 构造EMD-LSTM模型进行预测
    def emdLSTM(dataset):
        emd = EMD()
        IMFs = emd(dataset)  # 有6个IMFs
        print(len(IMFs))
        yhatResult = 0
        for n, imf in enumerate(IMFs):
            print("begin training IMF", n)
            tempDataSet = imf
            X_train, y_train, X_test, y_test = create(tempDataSet)
            if n == len(IMFs)-1:
                Epoch = 1000
                pred_y = create_LSTM(X_train, y_train, X_test, y_test, model_name='IMF'+str(n), Epochs=Epoch)
            else:
                Epoch = 100
                pred_y = create_LSTM(X_train, y_train, X_test, y_test, model_name='IMF'+str(n), Epochs=Epoch)
            yhatResult = pred_y + yhatResult
        return yhatResult

    # start = time.time()
    # predictions = emdLSTM(values)
    # end = time.time()
    # print('training time: ', end-start)


    def test_LSTM(train_x, train_y, test_x, test_y, model_name='LSTM', Epochs=100):
        # Reshaping the train_x and test_x data
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        # Creating the LSTM model
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(look_forward))

        # 加在模型参数
        model.load_weights('param//DLA' + file + model_name + '.h5')

        # Making predictions on the test data
        predictions = model.predict(test_x)

        # Calculating the root mean squared error
        rmse = sqrt(mean_squared_error(test_y, predictions))
        mae = mean_absolute_error(test_y, predictions)
        mse = mean_squared_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)

        print('Test MAE: %.3f' % mae)
        print('Test RMSE: %.3f' % rmse)
        print('Test MSE: %.3f' % mse)
        print('Test R2: %.3f' % r2)

        # model.save('param/DLA'+file+model_name+'.h5')

        return predictions


    def test_emdLSTM(dataset):
        emd = EMD()
        IMFs = emd(dataset)  # 有6个IMFs
        print(len(IMFs))
        yhatResult = 0
        for n, imf in enumerate(IMFs):
            print("begin training IMF", n)
            tempDataSet = imf
            X_train, y_train, X_test, y_test = create(tempDataSet)
            if n == len(IMFs) - 1:
                Epoch = 1000
                pred_y = test_LSTM(X_train, y_train, X_test, y_test, model_name='IMF' + str(n), Epochs=Epoch)
            else:
                Epoch = 100
                pred_y = test_LSTM(X_train, y_train, X_test, y_test, model_name='IMF' + str(n), Epochs=Epoch)
            yhatResult = pred_y + yhatResult
        return yhatResult

    start = time.time()
    predictions = test_emdLSTM(values)
    end = time.time()
    print('test time: ', end - start)

    #%%
    train_x, train_y, test_x, test_y = create(values)

    #%%

    # 计算MAE, MSE, RMSE和R2
    rmse = sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    mse = mean_squared_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)

    print('Test MAE: %.3f' % mae)
    print('Test RMSE: %.3f' % rmse)
    print('Test MSE: %.3f' % mse)
    print('Test R2: %.3f' % r2)
    #%%将误差写入文件
    with open('log/DLA'+file+'.txt', 'w') as f:
        f.write(file+'\n')
        f.write('Test MAE: %.3f\n' % mae)
        f.write('Test RMSE: %.3f\n' % rmse)
        f.write('Test MSE: %.3f\n' % mse)
        f.write('Test R2: %.3f\n' % r2)

    #%% 构造和test数据相同长度的predictionsiction数据
    test_prediction = []
    for i in range(len(test)-look_back):
        # 先判断有几个数
        flag = 0
        if i < look_forward-1:
            num = i+1
        elif i > len(test)-look_back-look_forward:
            num = len(test)-look_back-i
            flag = 1
        else:
            num = look_forward
        # 根据num添加数值
        if flag == 0:
            tmp = 0
            for j in range(num):
                tmp += predictions[i-j][j]
            tmp = tmp/num
            test_prediction.append(tmp)
        else:
            tmp = 0
            for j in range(num):
                tmp += predictions[i-look_forward+1+j][look_forward-1-j]
            tmp = tmp/num
            test_prediction.append(tmp)

    #%% 画图
    fontsize_tmp = 50
    import matplotlib.pyplot as plt

    # x = np.arange(1, len(values)+1)
    # plt.figure(figsize=(20, 10))
    # plt.plot(x, values, label='original')
    # plt.plot(x[train_size+look_back:len(values)+1], test_prediction, label='prediction')
    # plt.legend()
    # # plt.show()
    # plt.savefig('graph/桥面系挠度/LSTM.pdf')

    import matplotlib.dates as mdates

    # 设置绘图风格
    # plt.style.use('grayscale')

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
    plt.savefig('graph/DLA'+file[0:8] + '.pdf', bbox_inches='tight')