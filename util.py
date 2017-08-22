# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta
import cPickle as pickle
import pywt   # python 小波变换的包
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import LSTM, Input, Embedding, add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
def denoise(index_list, wavefunc, lv, m, n):
    # 按 m 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数
    coeff = pywt.wavedec(index_list, wavefunc, mode='sym', level=lv)
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0  # sgn函数
    # 去噪过程
    for i in range(m, n + 1):  # 选取小波系数层数为 m~n层，尺度系数不需要处理
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零
    # 重构
    denoised_index = pywt.waverec(coeff, wavefunc)
    return denoised_index

def nearest(df, column='forward'):
    df[column] = df[column].astype(np.int8)
    return df.sort_values(by=column)[0:1]

adj_dict = {}

def mape(ypredict, ytrue):
    return np.mean(np.abs(ypredict - ytrue) / ytrue)

def extract_dt(string):
    return datetime.strptime(string.split(',')[0].replace('[', ''), '%Y-%m-%d %H:%M:%S')

def load_dataset():
    dataset = pd.read_csv("/home/zhli7/guizhou/data/gy_contest_traveltime_training_data_second.txt", delimiter=';', header=0, names=['link', 'date', 'interval', 'travel_time'], parse_dates=['date'])
    dataset['dt'] = dataset.interval.map(extract_dt)
    return dataset

def load_sample():
    dataset = pd.read_csv("/home/zhli7/guizhou/data/gy_contest_link_traveltime_training_data_sample.txt", delimiter=';', header=None, names=['link', 'date', 'interval', 'travel_time'], parse_dates=['date'])
    dataset['dt'] = dataset.interval.map(extract_dt)
    dataset.index = dataset.dt
    dataset = dataset.sort_index()
    return dataset

import Queue
def search(link, adj):
    def search_one_step(link, direction):
        field = direction + '_links'
        try:
            links = getattr(adj[adj.link_ID==link], field).values[0]
            if links != '':
                links = links.split('#')
            else:
                links = []
        except:
            links = []
        #         print links
        return links

    def wide_search(root, direction):
        to_search = Queue.Queue()
        links = []
        done = {root: 1}
        start = search_one_step(root, direction)

        for l in start:
            if not done.has_key(l):
                to_search.put(l)

        while not to_search.empty():
            l = to_search.get(False)
            done[l] = 1
            links.append(l)
            for nt in search_one_step(l, direction):
                #                 print 'done ', done
                if not done.has_key(nt):
                    #                     print nt
                    to_search.put(nt)

        return links

    for d in ['in', 'out']:
        links = wide_search(link, d)
        print d, ' ', len(links), ' ', links

def load_data(dataframe, travel_time_set, history_mean, hour_minute_mean, hour_mean, start_date=None, end_date=None):
    dataset_time_index = []
    dataset_raw = []
    if not start_date:
        start_date = dataframe.index[0]
        start_date = datetime(start_date.year, start_date.month, start_date.day, 6)
    if not end_date:
        end_date = dataframe.index[-1]
        end_date = datetime(end_date.year, end_date.month, end_date.day, 6)

    for t in pd.date_range(start_date, end_date, freq='2min'):
        fill = 1
        if t.hour > 5:
            try:
                travel_time = dataframe.ix[t]['travel_time']
                fill = 0
            except:
                # print t
                try:
                    travel_time = history_mean[(t.isoweekday(), t.hour, t.minute)]
                except KeyError:
                    try:
                        travel_time = hour_minute_mean[(t.hour, t.minute)]
                    except KeyError:
                        travel_time = hour_mean[t.hour]
            try:
                travel_time_stat = travel_time_set[['mean', 'count', 'std', '50%']].values
                # 如果大于历史平均的两倍，认为是异常情况，，用历史平均代替
                if travel_time_stat[0] * 2 < travel_time:
                    travel_time = travel_time_stat[0]
            except:
                if len(dataset_raw) < 1:
                    travel_time = history_mean[(t.isoweekday(), t.hour, t.minute)]
                else:
                    travel_time = dataset_raw[-1][0]

            extra_feature = [fill, t.hour]
            feature = [travel_time] + extra_feature
            dataset_raw.append(feature)
            dataset_time_index.append(t.strftime('%Y%m%d%H%M%S'))

    return (dataset_raw, dataset_time_index)

# def create_dataset(length, dataset_raw, dataset_time_index, start, end, minutes_per_slot, look_back=1, predict_forwards=[1], test=False):
#     ds = np.array(dataset_raw)[:, 0:1]
#
#     scaler = MinMaxScaler(feature_range=(0,1))
#
#     ds[:, 0] = scaler.fit_transform((length / ds[:, 0]).reshape(-1, 1)).reshape(-1)
#     #     dataset[:, 0] = scaler.transform(dataset[:, 0].reshape(-1, 1)).reshape(-1)
#     dataX, dataY, timeIndexY = [], [], []
#     max_gap = float(max(predict_forwards)) + look_back
#
#     for predict_forward in predict_forwards:
#         for i in range(0, len(ds) - look_back - predict_forward + 1, slide):
#             if ds[i+look_back+predict_forward-1, 1] == 1:
#                 continue
#             # print ds[i:(i+look_back), :].shape, np.arange(look_back).reshape(-1, 1).shape
#             a = np.hstack((ds[i:(i+look_back), :], ((np.arange(look_back-1, -1, -1) + predict_forward) / 1).reshape(-1, 1)))
#
#             dataX.append(a)
#             #         dataY.append(ds[(i+look_back):(i+look_back+predict_forward), 0])
#             if predict_forward > 0:
#                 dataY.append(ds[(i+look_back+predict_forward-1):(i+look_back+predict_forward), 0])
#                 timeIndexY.append(dataset_time_index[(i+look_back+predict_forward-1):(i+look_back+predict_forward)])
#                 #         timeIndexY.append(dataset_time_index[(i+look_back):(i+look_back+predict_forward)])
#     return (np.array(dataX), np.array(dataY), scaler, timeIndexY,
#             pd.DataFrame(scaler.inverse_transform(ds[:, 0].reshape(-1, 1)), index=dataset_time_index, columns=["speed"]))

def create_all(dataframe, travel_time_set, look_back, predict_forward):
    dataset_raw, dataset_time_index = load_data(dataframe, travel_time_set)
    return create_dataset(dataset_raw, dataset_time_index, look_back, predict_forward)


def create_dataset(length, dataset_raw, dataset_time_index, start, end, slide, minutes_per_slot, look_back=1, predict_forwards=[1], test=False):
    ds = np.array(dataset_raw[:, :])
    data_start_time = datetime.strptime(dataset_time_index[0], "%Y%m%d%H%M%S")

    dataX, dataY, timeIndexY, forwards = [], [], [], []


    # 如果指定了起止时间
    if start and end:
        start_index = (start - data_start_time).days * 18 * 60 // minutes_per_slot \
                      + (start.hour - 6) * 60 // minutes_per_slot + start.minute // minutes_per_slot

        end_index = (end - data_start_time).days * 18 * 60 // minutes_per_slot \
                    + (end.hour - 6) * 60 // minutes_per_slot + end.minute // minutes_per_slot
    else:
        start_index = look_back + predict_forwards[-1]
        end_index = len(ds)

    # print start_index, end_index
    for i in range(int(start_index), int(end_index)+1):
        if i >= len(ds) and (not test):
            break
        # 不是构造测试集，并且目标值是填充值，跳过
        if (not test) and ds[i, 1] == 1:
            continue

        for predict_forward in predict_forwards:
            # 如果指定了开始时间，并且所用的训练数据在待预测范围内，跳过
            if (i - predict_forward) % (18 * 60 // minutes_per_slot) // (60 // minutes_per_slot) + 6 >= start.hour:
                continue
            else:
                a = np.hstack((ds[(i-look_back-predict_forward+1):(i-predict_forward+1), :], ((np.arange(look_back-1, -1, -1) + predict_forward)/1).reshape(-1, 1)))
                forwards.append(predict_forward)
                dataX.append(a)
                # 不是构造测试集时，要保存真值
                if not test:
                    dataY.append(ds[i:i+1, 0])

                timeIndexY.append(dataset_time_index[i:i+1])
                break
    return (np.array(dataX), np.array(dataY), forwards, timeIndexY)

def write_submission(dataframe, result_file):
    for index, row in dataframe.iterrows():
        link = index[0]
        t = datetime.strptime(index[1], "%Y%m%d%H%M%S")
        result_file.write("%s#%s#[%s,%s)#%f\n" % (link,
                                                  t.strftime("%Y-%m-%d"),
                                                  t.strftime("%Y-%m-%d %H:%M:%S"),
                                                  (t+timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
                                                  row['predict']))

def check_submission(fname, links):
    # 9377906286615510514#2016-06-30#[2016-06-30 08:32:00,2016-06-30 08:34:00)#5.538206
    result = pd.read_csv(fname, delimiter='#', names=['link', 'date', 'period', 'time'], index_col=["link", "period"])
    # assert len(result) == 30 * 30 * 132
    for link in links:
        print "checking link: %s" % (link)
        for d in range(1, 31):
            for t in pd.date_range(datetime(2017, 6, d, 8),
                                   datetime(2017, 6, d, 9), closed='left', freq='2min'):
                period = "[%s,%s)" % (t.strftime("%Y-%m-%d %H:%M:%S"),(t+timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"))
                result.ix[link, period]
        print "link: %s is complete" % (link)
        
        
def read_cache(fname):
    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    dataX, dataY, linkIdx, timestamp, forward, embedding = f['dataX'].value, f['dataY'].value, f['linkIdx'].value, f['timestamp'].value, f['forward'].value, f['embedding'].value    
    f.close()

    return dataX, dataY, linkIdx, timestamp, forward, embedding


def cache(fname, dataX, dataY, dataIdx, timestamp, forward, embedding):
    h5_train = h5py.File(fname, 'w')
    try:
        h5_train.create_dataset('num', data=len(dataX))
        h5_train.create_dataset('dataX', data=dataX)
        h5_train.create_dataset('dataY', data=dataY)
        h5_train.create_dataset('linkIdx', data=dataIdx, dtype='i')
        h5_train.create_dataset('timestamp', data=timestamp)
        h5_train.create_dataset('forward', data=forward)
        h5_train.create_dataset('embedding', data=embedding)
    except:
        h5_train.close()
        raise
        
def evaluate(all_predict, all_cvY, all_cvIdx, all_time, all_forward):
    cv_result = pd.DataFrame((np.hstack((all_predict, all_cvY, 
                                         np.array(all_time).reshape(-1, 1), 
                                         np.array(all_forward).reshape(-1, 1),
                                        np.array(all_cvIdx).reshape(-1, 1)))), 
                 columns=['predict', 'real', 'time', 'forward', 'link'])
    cv_result['predict'] = cv_result.predict.astype(np.float32)
    cv_result['real'] = cv_result.real.astype(np.float32)
    cv_nearest = cv_result.groupby(['link', 'time']).apply(nearest)

    predict = cv_nearest['predict'].values
    real = cv_nearest['real'].values
    cv_mape = mape(predict, real)
    print 'cv mape: {}'.format(cv_mape)
    return cv_mape

# def create_train(links, f, hypername, info, link2idx, link_embeddings, g_look_back, g_predict_forward, slide, denoised=False, minutes_per_slot=2):
#     trainX, trainY = [], []
#     trainIdx = []
#     trainTimestamp = []
#     scalers = {}
#     all_forward = []
#     trainEmbeddings = []
#     for link in links:
#         train_ds, train_ts = f[link]['train']['data'].value, f[link]['train']['time'].value
#         length = info[info.link_ID==link]['length'].values[0]
#     #     stmatrix = STMatrix(train_ds, train_ts, 
#     #                     T=720, CheckComplete=False, start=1)
#         scaler = MinMaxScaler(feature_range=(0,1))

#         train_ds[:, 0] = scaler.fit_transform((length / train_ds[:, 0]).reshape(-1, 1)).reshape(-1)
#         # train_ds = train_ds[:, [0, 2]]
#         # train_ds[:, 0] = length / train_ds[:, 0]
#         if denoised:
#             train_ds[:, 0] = denoise(train_ds[:, 0], 'db3', 3, 2, 3)[:-1]


#     #     for forward in g_predict_forward:
#         # length, dataset_raw, dataset_time_index, start, end, slide, minutes_per_slot, look_back=1, predict_forwards=[1], test=False
#         trainX_l, trainY_l, forwards, trainYTimeIndex = \
#             create_dataset(length, train_ds, train_ts, None, None, slide, minutes_per_slot, g_look_back, g_predict_forward, False)
#     #             trainX_l, XP, XT, trainY_l, trainYTimeIndex = stmatrix.create_dataset(forward=forward, len_closeness=g_look_back, len_trend=0, TrendInterval=7, len_period=0, PeriodInterval=1)
#         # trainX_l = trainX_l[:, :, :2]
#         trainX.append(trainX_l)
#         trainY.append(trainY_l)
#         trainTimestamp.append(trainYTimeIndex)
#         trainIdx.append(np.ones((trainX_l.shape[0], 1)) * link2idx[link])
#         all_forward += forwards
#         trainEmbeddings.append(np.array(link_embeddings[link]) * np.ones((trainX_l.shape[0], 1))) 

#     trainX = np.vstack(trainX)
#     trainY = np.vstack(trainY)
#     trainIdx = np.vstack(trainIdx).astype(np.int8)
#     trainTimestamp = np.vstack(trainTimestamp).astype(np.str)
#     trainEmbeddings = np.vstack(trainEmbeddings)
#     all_forward = np.array(all_forward)
#     fname = 'CACHE/{}_train.h5'.format(hypername)
#     cache(fname, trainX, trainY, trainIdx, trainTimestamp, all_forward, trainEmbeddings)
#     return fname
#     # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))

def create_target_set(dset, links, f, hypername, info, link2idx, link_embeddings, g_look_back, g_predict_forward, slide, daterange, is_test, denoised=False, minutes_per_slot=2):
    all_predict = []
    all_cvY = []
    all_cvX = []
    all_cvIdx = []
    cvEmbeddings = []
    # all_time = np.empty((0,0))
    all_time = []
    all_forward = []
    all_links = []
    for link in links:
        cv_ds, cv_ts = f[link][dset]['data'].value, f[link][dset]['time'].value
        length = info[info.link_ID==link]['length'].values[0]
        scaler = MinMaxScaler(feature_range=(0,1))

        cv_ds[:, 0] = scaler.fit_transform((length / cv_ds[:, 0]).reshape(-1, 1)).reshape(-1)
        # cv_ds = cv_ds[:, [0, 2]]
        # cv_ds[:, 0] = length / cv_ds[:, 0]
        for day in daterange:
        # for day in range(25, 30):
            for hour in range(8, 19, 3):
                predict_window_start = datetime(day.year, day.month, day.day, hour, 0)
                predict_window_end = predict_window_start + timedelta(minutes=59)

                if denoised:
                    cv_ds[:, 0] = denoise(cv_ds[:, 0], 'db3', 4, 1, 4)[:-1]
            #     train_window_start = datetime(2016, 5, day, hour, 0) - timedelta(hours=2)
            #     train_window_end = train_window_start + timedelta(hours=1, minutes=59)

                cvX, cvY, forwards, cvYTime = create_dataset(length, cv_ds, 
                                                                      cv_ts, 
                                                                      predict_window_start, 
                                                                      predict_window_end, 
                                                                      1,
                                                                      minutes_per_slot, 
                                                                      g_look_back, 
                                                                      g_predict_forward, is_test)
                if len(cvX) < 1:
                    continue

                cvIdx = np.ones((cvX.shape[0], 1)) * link2idx[link]
        #         cvX = np.reshape(cvX, (cvX.shape[0], cvX.shape[1], cvX.shape[2]))
        #         cvPredict = model.predict([cvX, cvEmbedding])
        #         cvPredict = scaler.inverse_transform(cvPredict)
        #         cvY = scaler.inverse_transform(cvY)

        #         all_predict.append(cvPredict)
                # cvX = cvX[:, :, :2]
                all_cvX.append(cvX)
                all_cvY.append(cvY)
                all_cvIdx.append(cvIdx)

        #         all_links += [link] * len(cvX)
                all_forward = all_forward + forwards
                all_time = all_time + list(np.reshape(cvYTime, -1))
                cvEmbeddings.append(np.array(link_embeddings[link]) * np.ones((cvX.shape[0], 1))) 
    all_cvX = np.vstack(all_cvX)
    all_cvY = np.vstack(all_cvY)
    all_time = np.asarray(all_time)
    all_cvIdx = np.vstack(all_cvIdx)
    all_forward = np.array(all_forward)
    cvEmbeddings = np.vstack(cvEmbeddings)
    fname = 'CACHE/{}_{}.h5'.format(dset, hypername)
    cache(fname, all_cvX, all_cvY, all_cvIdx, all_time, all_forward, cvEmbeddings)
    return fname
# print "cv MAPE: %f" % (mape(all_predict, all_Y))
    
def build_model(trainX, links):
    hidden_neurons = 64
    # loss_function = 'mean_absolute_percentage_error'
    loss_function = 'mae'
    dropout = 0.02
    inner_hidden_neurons = 64
    dropout_inner = 0.02
    out_neurons = 1
    lstm_input = Input(shape=(trainX.shape[1], trainX.shape[2]), name='lstm_input')
    link_input = Input(shape=(1,), name='link_input')

    link_embedding = Embedding(output_dim=4, input_dim=len(links), name='link_embedding')(link_input)
    link_embedding = Flatten()(link_embedding)
    
    deepwalk_input = Input(shape=(8,), name='deepwalk')
    forward_input = Input(shape=(1,), name='foward')
#     deepwalk_embedding = Dense(8)(deepwalk_input)
#     deepwalk_embedding = Flatten()(deepwalk_embedding)
    
    # model.add(LSTM(4, batch_input_shape=(batch_size, look_back, trainX.shape[2]), activation='tanh', dropout_U=0.05, stateful=True, return_sequences=True))
    # model.add(LSTM(8, batch_input_shape=(batch_size, look_back, trainX.shape[2]), activation='tanh', dropout_U=0.05, stateful=True))
    # model.add(Dense(1))
    # for i in range(100):
    #     model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
    #     model.reset_states()
    in_neurons = trainX.shape[2]
    gpu_cpu = 'cpu'
    # lstm1 = LSTM(output_dim=hidden_neurons, input_dim=in_neurons, 
    #                   return_sequences=True, init='uniform',
    #                    consume_less=gpu_cpu, name='lstm1')(lstm_input)
    # model.add(Dropout(dropout))

    dense_input = inner_hidden_neurons
    # model.add(LSTM(output_dim=inner_hidden_neurons, input_dim=hidden_neurons, return_sequences=True, consume_less=gpu_cpu))
    # model.add(Dropout(dropout_inner))
    lstm2 = LSTM(input_dim=in_neurons, output_dim=dense_input, name='lstm1', return_sequences=False)(lstm_input)
    # model.add(Dropout(dropout_inner))
    lstm_output = Activation('relu')(lstm2)
    lstm_model = Dense(output_dim=dense_input, input_dim=dense_input)(lstm_output)
    lstm_model = Activation('relu')(lstm_model)
    lstm_model = Dense(output_dim=dense_input, input_dim=dense_input)(lstm_model)
    lstm_model = Dropout(0.5)(lstm_model)
    lstm_model = Activation('relu')(lstm_model)
    residual = Dense(output_dim=dense_input, input_dim=dense_input)(lstm_output)
    residual = Activation('relu')(residual)
    # lstm_output = add([lstm_model, residual])
    lstm_output = lstm_model

    # merge_output = concatenate([lstm_output, link_embedding, deepwalk_input, forward_input])
    merge_output = concatenate([lstm_output, link_embedding, deepwalk_input])
    merge_output = Dense(16, name='merge')(merge_output)
    merge_output = Dense(32, name='merge2')(merge_output)
    merge_output = Dense(24, name='merge3')(merge_output)
    
    output = Dense(1, name='output')(merge_output)

    # model = Model([lstm_input, link_input, deepwalk_input, forward_input], output)
    model = Model([lstm_input, link_input, deepwalk_input], output)
    # model.add(Activation('linear'))
    model.compile(loss=loss_function, optimizer='adam', metrics=['mae'])
    model.lr = 0.001
    return model
# weights = cv_result.groupby(['link', 'forward']).apply(lambda g: 1.0 / mape(g['predict'], g['real']) ** 2)

# weights_df = pd.DataFrame(weights, columns=['weight'])

# cv_result = pd.merge(cv_result, weights_df, left_on=['link', 'forward'], right_index=True)

# mape(cv_result.predict.values, cv_result.real.values)

# weighted = cv_result.groupby(['link','time']).apply(lambda g: \
#                                        np.average(g[['predict', 'real']], axis=0, weights=g['weight']))

# predict = np.array([weighted.values[i][0] for i in range(len(weighted))])
# real = np.array([weighted.values[i][1] for i in range(len(weighted))])

# mape(predict, real)