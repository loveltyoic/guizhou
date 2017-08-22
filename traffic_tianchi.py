
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from util import *

import h5py

adj = pd.read_csv("data/gy_contest_link_top.txt", delimiter=';', dtype={'in_links': np.str, 'out_links': str})
adj = adj.fillna('')

info = pd.read_csv("data/gy_contest_link_info.txt", delimiter=';')

info.head()

# stage = 'sample'
stage = 'data'
f = h5py.File(stage + '/travel_time.hdf5', 'r')

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, Input, Embedding, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


minutes_per_slot = 2

links = f.keys()
links[0]
info[info.link_ID==links[0]]['length'].values[0]

link2idx = dict(zip(links, range(len(links))))

trainX, trainY = None, None
trainEmbedding = None
for link in links:
    train_ds, train_ts = f[link]['train']['data'].value, f[link]['train']['time'].value
    length = info[info.link_ID==link]['length'].values[0]
    g_look_back = 4
    g_predict_forward = range(1, 31)
    if trainX != None:
        trainX_l, trainY_l, trainScaler, trainYTimeIndex, trainDF = create_dataset(length, train_ds, train_ts, g_look_back, g_predict_forward)
        trainEmbedding = np.vstack((trainEmbedding, np.ones((trainX_l.shape[0], 1)) * link2idx[link]))
        trainX = np.vstack((trainX, trainX_l))
        trainY = np.vstack((trainY, trainY_l))
    else:
        trainX, trainY, trainScaler, trainYTimeIndex, trainDF = create_dataset(length, train_ds, train_ts, g_look_back, g_predict_forward)
        trainEmbedding =np.ones((trainX.shape[0], 1)) * link2idx[link]

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))

hidden_neurons = 32
loss_function = 'mse'
batch_size = 100
dropout = 0.02
inner_hidden_neurons = 64
dropout_inner = 0.02
out_neurons = 1
lstm_input = Input(shape=(trainX.shape[1], trainX.shape[2]), name='lstm_input')
link_input = Input(shape=(1,), name='link_input')
link_embedding = Embedding(output_dim=8, input_dim=len(links), name='link_embedding')(link_input)
link_embedding = Flatten()(link_embedding)

# model.add(LSTM(4, batch_input_shape=(batch_size, look_back, trainX.shape[2]), activation='tanh', dropout_U=0.05, stateful=True, return_sequences=True))
# model.add(LSTM(8, batch_input_shape=(batch_size, look_back, trainX.shape[2]), activation='tanh', dropout_U=0.05, stateful=True))
# model.add(Dense(1))
# for i in range(100):
#     model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
#     model.reset_states()
in_neurons = trainX.shape[2]
gpu_cpu = 'cpu'
lstm1 = LSTM(output_dim=hidden_neurons, input_dim=in_neurons,
             return_sequences=True, init='uniform',
             consume_less=gpu_cpu, name='lstm1')(lstm_input)
# model.add(Dropout(dropout))

dense_input = inner_hidden_neurons
# model.add(LSTM(output_dim=inner_hidden_neurons, input_dim=hidden_neurons, return_sequences=True, consume_less=gpu_cpu))
# model.add(Dropout(dropout_inner))
lstm2 = LSTM(input_dim=hidden_neurons, output_dim=dense_input, name='lstm2', return_sequences=False)(lstm1)
# model.add(Dropout(dropout_inner))
lstm_output = Activation('relu')(lstm2)
lstm_model = Dense(output_dim=dense_input*2, input_dim=dense_input)(lstm_output)
residual = Dense(output_dim=dense_input*2, input_dim=dense_input*2)(lstm_output)
residual = Activation('relu')(residual)
lstm_output = merge([lstm_model, residual], mode='sum')

merge_output = merge([lstm_output, link_embedding], mode='concat')
output = Dense(1, name='output')(merge_output)

model = Model([lstm_input, link_input], output)
# model.add(Activation('linear'))
early_stopping = EarlyStopping(monitor='val_rmse', patience=2, mode='min')
model_checkpoint = ModelCheckpoint(
    'MODEL/best.h5', monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
tensorboard = TensorBoard()
model.compile(loss='mse', optimizer='adam')
model.lr = 0.001
# model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=1, shuffle=True)

from keras.utils import plot_model
plot_model(model, to_file='lstm.png', show_shapes=True)

model.fit([trainX, trainEmbedding], trainY, epochs=5,
          batch_size=batch_size, verbose=1, shuffle=True,
          callbacks=[early_stopping, model_checkpoint, tensorboard])

# testScore = mape(testPredict[:, 0], testY)
# print('Test Score %.2f MAPE' % (testScore))

model.save('MODEL/forward30.h5')

all_predict = np.empty((0,0))
all_Y = np.empty((0,0))
# all_time = np.empty((0,0))
all_time = []
all_forward = []
all_links = []
for link in links:
    cv_ds, cv_ts = f[link]['cv']['data'].value, f[link]['cv']['time'].value
    length = info[info.link_ID==link]['length'].values[0]
    for day in range(25, 30):
        predict_window_start = datetime(2016, 5, day, 8, 0)
        predict_window_end = predict_window_start + timedelta(minutes=59)


        #     train_window_start = datetime(2016, 5, day, hour, 0) - timedelta(hours=2)
        #     train_window_end = train_window_start + timedelta(hours=1, minutes=59)

        cvX, cvY, forwards, cvScaler, cvYTime, cvDF = create_cv_dataset(length, cv_ds,
                                                                        cv_ts,
                                                                        predict_window_start,
                                                                        predict_window_end,
                                                                        minutes_per_slot,
                                                                        g_look_back,
                                                                        range(1, 31))
        cvEmbedding = np.ones((cvX.shape[0], 1)) * link2idx[link]
        cvX = np.reshape(cvX, (cvX.shape[0], cvX.shape[1], cvX.shape[2]))
        cvPredict = model.predict([cvX, cvEmbedding])
        cvPredict = cvScaler.inverse_transform(cvPredict)
        cvY = cvScaler.inverse_transform(cvY)
        if all_predict.any():
            all_predict = np.vstack((all_predict, cvPredict))
            all_Y = np.vstack((all_Y, cvY))
            #         all_time = np.vstack((all_time, cvYTime))
        else:
            all_predict = cvPredict
            all_Y = cvY
            #         all_time = cvYTime
        all_links += [link] * len(cvX)
        all_forward = all_forward + forwards
        all_time = all_time + list(np.reshape(cvYTime, -1))

# print "cv MAPE: %f" % (mape(all_predict, all_Y))

cv_result = pd.DataFrame((np.hstack((all_predict, all_Y,
                                     np.array(all_time).reshape(-1, 1),
                                     np.array(all_forward).reshape(-1, 1),
                                     np.array(all_links).reshape(-1, 1)))),
                         columns=['predict', 'real', 'time', 'forward', 'link'])
cv_result['predict'] = cv_result.predict.astype(np.float32)
cv_result['real'] = cv_result.real.astype(np.float32)

cv_result

weights = cv_result.groupby(['link', 'forward']).apply(lambda g: 1.0 / mape(g['predict'], g['real']) ** 2)

weights_df = pd.DataFrame(weights, columns=['weight'])

cv_result = pd.merge(cv_result, weights_df, left_on=['link', 'forward'], right_index=True)

mape(cv_result.predict.values, cv_result.real.values)

cv_result

weighted = cv_result.groupby(['link','time']).apply(lambda g: \
                                                        np.average(g[['predict', 'real']], axis=0, weights=g['weight']))

predict = np.array([weighted.values[i][0] for i in range(len(weighted))])
real = np.array([weighted.values[i][1] for i in range(len(weighted))])

mape(predict, real)

cv_nearest = cv_result.groupby(['link', 'time']).apply(nearest)

predict = cv_nearest['predict'].values
real = cv_nearest['real'].values
mape(predict, real)

cv_result['forward'] = cv_result.forward.astype(np.int8)

# pd.DataFrame(all_predict, index=all_time).resample('20min').mean().plot(label='predict', ax=ax)
# pd.DataFrame(all_Y, index=all_time).resample('20min').mean().plot(ax=ax)

# ticks = [i * real.shape[0] / 10 for i in range(10)]
# ax.set_xticks(ticks)
# ax.set_xticklabels([cv_nearest.index[t][0] for t in ticks])

denoised_data = np.reshape(denoise(cv_ds[:100, 0], 'db3', 3, 2, 3), (-1, 1))

all_predict = np.empty((0,0))
all_Y = np.empty((0,0))
# all_time = np.empty((0,0))
all_time = []
all_test_forwards = []
all_links = []
for link in links:
    test_ds, test_ts = f[link]['test']['data'].value, f[link]['test']['time'].value
    length = info[info.link_ID==link]['length'].values[0]
    for day in range(1, 31):
        predict_window_start = datetime(2016, 6, day, 8, 0)
        predict_window_end = predict_window_start + timedelta(minutes=59)


        #     train_window_start = datetime(2016, 5, day, hour, 0) - timedelta(hours=2)
        #     train_window_end = train_window_start + timedelta(hours=1, minutes=59)

        testX, testY, test_forwards, testScaler, testYTime, testDF = create_cv_dataset(length, test_ds,
                                                                                       test_ts,
                                                                                       predict_window_start,
                                                                                       predict_window_end,
                                                                                       minutes_per_slot,
                                                                                       g_look_back,
                                                                                       range(1, 31),True)

        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
        testEmbedding = np.ones((testX.shape[0], 1)) * link2idx[link]
        testPredict = model.predict([testX, testEmbedding])
        testPredict = testScaler.inverse_transform(testPredict)
        testPredict = length / testPredict

        if all_predict.any():
            all_predict = np.vstack((all_predict, testPredict))
            #         all_time = np.vstack((all_time, cvYTime))
        else:
            all_predict = testPredict
            #         all_time = cvYTime
        all_test_forwards += test_forwards
        all_time = all_time + list(np.reshape(testYTime, -1))
        all_links += [link] * len(testX)
# print "cv MAPE: %f" % (mape(all_predict, all_Y))

len(all_time), all_predict.shape, np.array(all_test_forwards).reshape(-1, 1).shape

test_result = pd.DataFrame((np.hstack((all_predict, np.array(all_time).reshape(-1, 1),
                                       np.array(all_test_forwards).reshape(-1, 1),
                                       np.array(all_links).reshape(-1, 1)))),
                           columns=['predict', 'time', 'forward', 'link'])
# test_result.index=test_result.time
test_result['predict'] = test_result.predict.astype(np.float32)

test_nearest = test_result.groupby(['link', 'time']).apply(nearest)

submission_name = "credo_%s.txt" % datetime.now().strftime('%Y%m%d')
write_submission(test_nearest, file(submission_name, 'w'))


