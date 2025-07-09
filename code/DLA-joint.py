#%%
import pandas as pd
import numpy as np
from config import get_config
from PyEMD import EMD
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import EarlyStopping

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

import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Add
from keras.models import Model

# %% 构造train_x, train_y, test_x和test_y，使用24步预测6步
def create_dataset(dataset, look_back=1, look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward + 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back + look_forward])
    return np.array(dataX), np.array(dataY)

def build_joint_emd_lstm(imf_num, look_back, look_forward):
    inputs = []
    imf_outputs = []
    for i in range(imf_num):
        inp = Input(shape=(1, look_back), name=f'input_imf_{i}')
        x = LSTM(4)(inp)
        out = Dense(look_forward, name=f'output_imf_{i}')(x)
        inputs.append(inp)
        imf_outputs.append(out)
    # 叠加所有imf的输出，得到整体预测
    total_output = Add(name='total_output')(imf_outputs)
    model = Model(inputs=inputs, outputs=imf_outputs + [total_output])
    return model

def emdLSTM_joint(dataset, look_back, look_forward, alpha=1.0, beta=1.0, epochs=100):
    # 1. EMD分解
    emd = EMD()
    IMFs = emd(dataset)
    imf_num = len(IMFs)
    print(f"IMF数量: {imf_num}")

    # 2. 构造每个IMF的滑窗数据
    Xs_train, Ys_train = [], []
    Xs_test, Ys_test = [], []
    total_len = len(dataset)
    train_size = int(total_len * 0.8)
    for imf in IMFs:
        train = imf[:train_size]
        test = imf[train_size:]
        X_train, Y_train = create_dataset(train, look_back, look_forward)
        X_test, Y_test = create_dataset(test, look_back, look_forward)
        # reshape为LSTM输入格式
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        Xs_train.append(X_train)
        Ys_train.append(Y_train)
        Xs_test.append(X_test)
        Ys_test.append(Y_test)

    # 3. 构造整体输出（所有IMF之和）
    # 训练集
    Y_total_train = np.sum([y for y in Ys_train], axis=0)
    # 测试集
    Y_total_test = np.sum([y for y in Ys_test], axis=0)
    Ys_train_joint = Ys_train + [Y_total_train]
    Ys_test_joint = Ys_test + [Y_total_test]

    # 4. 构建联合模型
    model = build_joint_emd_lstm(imf_num, look_back, look_forward)

    # 计算每个IMF和整体的标准差
    imf_stds = [np.std(y) for y in Ys_train]
    total_std = np.std(Y_total_train)

    losses = {}
    loss_weights = {}
    for i in range(imf_num):
        losses[f'output_imf_{i}'] = tf.keras.losses.MeanSquaredError()
        # 归一化loss权重
        loss_weights[f'output_imf_{i}'] = alpha / (imf_stds[i] + 1e-8)
    losses['total_output'] = tf.keras.losses.MeanSquaredError()
    loss_weights['total_output'] = beta / (total_std + 1e-8)

    # # 用多输出loss和loss_weights
    # losses = {}
    # for i in range(imf_num):
    #     losses[f'output_imf_{i}'] = tf.keras.losses.MeanSquaredError()
    # losses['total_output'] = tf.keras.losses.MeanSquaredError()

    # loss_weights = {}
    # for i in range(imf_num):
    #     loss_weights[f'output_imf_{i}'] = alpha
    # loss_weights['total_output'] = beta

    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)

    # 5. 训练
    early_stopping = EarlyStopping(monitor='loss', mode='min', patience=80, verbose=1)
    model.fit(Xs_train, Ys_train_joint, 
              epochs=epochs, 
              batch_size=32, 
              validation_data=(Xs_test, Ys_test_joint),
              callbacks=[early_stopping],
              verbose=0)  # 设置为0以隐藏训练过程的输出
    
    # 冻结前 N-1 个 IMF
    for i in range(imf_num-1):
        model.get_layer(f'input_imf_{i}').trainable = False
        model.get_layer(f'output_imf_{i}').trainable = False

    # 重新编译模型（冻结/解冻后必须重新编译）
    model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)

    # 训练900 epoch
    model.fit(Xs_train, Ys_train_joint, epochs=900, batch_size=32, validation_data=(Xs_test, Ys_test_joint), callbacks=[early_stopping], verbose=0)
    
    # 保存模型
    model.save(f'param/DLA-joint'+file[0:8] + '.keras')

    # 6. 预测
    preds = model.predict(Xs_test)
    # preds是列表：[imf1_pred, ..., imfN_pred, total_pred]
    total_pred = preds[-1]
    return total_pred, Ys_test_joint[-1], preds[:-1], Ys_test  # 新增返回各IMF预测和真实

# 用法示例
for file in filelist:
    print('node: ', file)
    values = readData(filepath+file)

    # 数据划分
    train_size = int(len(values) * 0.8)
    train = values[:train_size]
    test = values[train_size:]  # <--- 这里定义 test

    pred, y_true, imf_preds, imf_ys = emdLSTM_joint(values, look_back, look_forward, alpha=1.0, beta=1.0, epochs=100)
    # pred, y_true shape: (样本数, look_forward)

    # 计算MAE, MSE, RMSE和R2
    rmse = sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    mse = mean_squared_error(y_true, pred)
    r2 = r2_score(y_true, pred)

    print('Test MAE: %.3f' % mae)
    print('Test RMSE: %.3f' % rmse)
    print('Test MSE: %.3f' % mse)
    print('Test R2: %.3f' % r2)

    # 计算并打印每个IMF的MAE
    for i, (imf_pred, imf_y) in enumerate(zip(imf_preds, imf_ys)):
        imf_mae = mean_absolute_error(imf_y, imf_pred)
        print(f'IMF{i+1} MAE: {imf_mae:.3f}')

    #%%将误差写入文件
    with open('log/DLA-joint'+file+'.txt', 'w') as f:
        f.write(file+'\n')
        f.write('Test MAE: %.3f\n' % mae)
        f.write('Test RMSE: %.3f\n' % rmse)
        f.write('Test MSE: %.3f\n' % mse)
        f.write('Test R2: %.3f\n' % r2)

    #%% 构造和test数据相同长度的prediction数据
    # 假设 pred 是一个二维 numpy 数组
    pred = np.array(pred)

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
        values = [pred[row, col] for row, col in indices]
        avg = np.mean(values)
        test_prediction.append(avg)

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
    plt.savefig('graph/DLA-joint'+file[0:8] + '.pdf', bbox_inches='tight')