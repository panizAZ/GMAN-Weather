import numpy as np
import pandas as pd
import h5py


# This file handles everything before the model:
# Logging (log_string)
# Metrics computation (metric)
# Convert sequences to supervised learning format (seq2instance)
# Load and preprocess data:
#       Split train/val/test
#       Normalize
#       Load spatial and temporal embeddings


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
# Computes evaluation metrics for traffic prediction:
# MAE (Mean Absolute Error)
# RMSE (Root Mean Squared Error)
# MAPE (Mean Absolute Percentage Error)
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def loadData(args):
    # =========================
    # Load fused traffic+weather data (HDF5)
    # =========================
    with h5py.File(args.traffic_file, 'r') as f:
        data = f['data'][:]  # shape: (T, N, C)
    num_step, num_nodes, num_features = data.shape

    Traffic = data.reshape(num_step, -1)  # (T, N*C)

    # train/val/test
    num_step = data.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]


    # reshapes (time, N) into (num_samples, P, N, W+1) for X and (num_samples, Q, N, W+1) for Y.
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)


    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std

    # spatial embedding
    # reads precomputed node embeddings (from Node2Vec).
    f = open(args.SE_file, mode = 'r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape = (N, dims), dtype = np.float32)
    for line in lines[1 :]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1 :]

    # Reconstruct time index
    start_time = pd.Timestamp('2017-01-01 00:00:00')  # adjust if your dataset is different
    time_index = pd.date_range(start=start_time, periods=num_step, freq='5min')

    # Temporal embedding
    dayofweek = np.reshape(time_index.weekday, newshape=(-1, 1))
    timeofday = (time_index.hour * 3600 + time_index.minute * 60 + time_index.second) // 300  # 300s = 5min
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)

    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

    ####
    trainX = trainX[:5000]
    trainTE = trainTE[:5000]
    trainY = trainY[:5000]

    valX = valX[:1000]
    valTE = valTE[:1000]
    valY = valY[:1000]

    testX = testX[:1000]
    testTE = testTE[:1000]
    testY = testY[:1000]
    
    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)
