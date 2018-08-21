from flask import Flask, jsonify, request
from flask import Response

import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)

train_features_filename = 'train.h5'
# hyper parameters
n_estimators = 100
max_samples = 100
contamination = 0.2
max_features = 3
rand = np.random.RandomState(42)

@app.route('/train1', methods=['POST'])
def train_normal():
    # retrieve data from request
    cpu, network, time = preprocessData(str(request.data))
    # retrieve train features
    train_features = generateTrainFeatures(cpu, network)
    # train model
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features=max_features, random_state=rand)
    model.fit(train_features)
    # save model
    joblib.dump(model, 'model.pkl')
    # save features
    storeTrainFeatures(train_features)
    # plot model
    plot_data_in_different_angle(train_features, None, None)

    return "OK", 200

@app.route('/train2', methods=['POST'])
def train_with_business_grow():
    train_features = None
    # loading data from request
    cpu, network, time = preprocessData(str(request.data))
    # looping generate plot imsages
    for i in range(1, 17):
        train_features = generateTrainFeatures(cpu + i, network + i)
        plot_data(train_features, idx=i)
    # train model
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features=max_features, random_state=rand)
    model.fit(train_features)
    # save model
    joblib.dump(model, 'model.pkl')
    # save features
    storeTrainFeatures(train_features)

    return "OK", 200

@app.route('/benchmark', methods=['POST'])
def benchmark():
    # loading model
    model = joblib.load('model.pkl')
    if model == None:
        return "Please train model first!"
    # generate benchmark test features
    test_features = generateTestFeatures()
    # do predict
    predictions = predict(test_features)
    # plot result
    plot_data_in_different_angle(loadTrainFeatures(), test_features, predictions)
    # generate return result
    result = ''
    for pred in predictions:
        result += str(pred) + ","
    return result, 200

@app.route('/predict', methods=['POST'])
def customize_predict():
    # loading pre-trained model
    model = joblib.load('model.pkl')
    if model == None:
        return "Please train model first!!"
    # retreive data from request
    cpu, network, time = preprocessData(str(request.data))
    # generate test features
    test_features = generateSpecifyTestFeatures(cpu, network, time)
    # do predict
    predictions = predict(test_features)
    # plot data
    plot_data(loadTrainFeatures(), test_features=test_features, predictions=predictions)

    return str(predictions[0]), 200

def generateTestFeatures():
    test_features = []
    # normal data at 3:00
    #test_features.append(np.array([15, 15, retrieveHourOfDay(3)]))
    # anormaly data at 03:00
    #test_features.append(np.array([35, 35, retrieveHourOfDay(3)]))

    # normal data at 8:00
    test_features.append(np.array([25, 25, retrieveHourOfDay(8)]))
    # anormaly data at 8:00
    test_features.append(np.array([40, 40, retrieveHourOfDay(8)]))

    # normal data at 12:00
    test_features.append(np.array([35, 35, retrieveHourOfDay(12)]))
    # anormaly data at 12:00
    test_features.append(np.array([50, 50, retrieveHourOfDay(12)]))

    # normal data at 20:00
    test_features.append(np.array([25, 25, retrieveHourOfDay(20)]))
    # anormaly data at 20:00
    test_features.append(np.array([40, 40, retrieveHourOfDay(20)]))

    # normal data at 23:00
    #test_features.append(np.array([15, 15, retrieveHourOfDay(23)]))
    # anormaly data at 23:00
    #test_features.append(np.array([35, 35, retrieveHourOfDay(23)]))

    test_features = np.reshape(test_features, (6, 3))

    return test_features

def generateTrainFeatures(cpu=10, network=10):
    # idle data from 01:00 to 07:00
    cpu_load_1 = generateCPULoadOrNetwork(cpu, cpu + 10, 800)
    net_work_1 = generateCPULoadOrNetwork(network, network + 10, 800)
    hourofday_1 = generateTimeRange(0, 7, 100)

    # growing data from 07:00 to 10:00
    cpu_load_2 = generateCPULoadOrNetwork(cpu + 10, cpu + 20, 400)
    net_work_2 = generateCPULoadOrNetwork(network + 10, network + 20, 400)
    hourofday_2 = generateTimeRange(7, 10, 100)

    # busy data from 10:00 to 19:00
    cpu_load_3 = generateCPULoadOrNetwork(cpu + 20, cpu + 30, 1000)
    net_work_3 = generateCPULoadOrNetwork(network + 20, network + 30, 1000)
    hourofday_3 = generateTimeRange(10, 19, 100)

    # decrease data from 19:00 to 21:00
    cpu_load_4 = generateCPULoadOrNetwork(cpu + 10, cpu + 20, 300)
    net_work_4 = generateCPULoadOrNetwork(network + 10, network + 20, 300)
    hourofday_4 = generateTimeRange(19, 21, 100)

    # idle data from 21:00 to 24:00
    cpu_load_5 = generateCPULoadOrNetwork(cpu, cpu + 10, 400)
    net_work_5 = generateCPULoadOrNetwork(network, network + 10, 400)
    hourofday_5 = generateTimeRange(21, 24, 100)

    # concate busy, growing, decrease and idle data
    cpu_load = np.r_[cpu_load_1, cpu_load_2, cpu_load_3, cpu_load_4, cpu_load_5]
    net_work = np.r_[net_work_1, net_work_2, net_work_3, net_work_4, net_work_5]
    hourofday = np.reshape(np.r_[hourofday_1, hourofday_2, hourofday_3, hourofday_4, hourofday_5], (2900, 1))
    # stack data as feature matrix
    train_features = np.hstack((cpu_load, net_work, hourofday))
    return train_features


def storeTrainFeatures(train_features):
    # delete file if already exist
    if os.path.isfile(train_features_filename):
        os.remove(train_features_filename)
    with h5py.File(train_features_filename) as h:
        h.create_dataset("train", data=train_features)

def loadTrainFeatures():
    train_features = None
    with h5py.File(train_features_filename, 'r') as h:
        train_features = np.array(h['train'])
    return train_features

def preprocessData(data):
    data = data.replace("\'", "")
    data = data.replace("b", "")
    data = json.loads(data)
    cpu = data['cpu']
    network = data['network']
    time = data['time']
    return cpu, network, time

def generateSpecifyTestFeatures(cpu, network, time):
    test_features = []
    test_features.append(np.array([cpu, network, retrieveHourOfDay(time)]))
    return np.reshape(test_features, (1, 3))

def predict(features):
    model = joblib.load('model.pkl')
    prediction = model.predict(features)
    return prediction


def retrieveColor(pred):
    if pred == -1:
        return 'r'
    else:
        return 'g'

def plot_data(normal_features, test_features=None, predictions=None, angle=None, idx=None):
    # create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('TIME')
    ax.set_ylabel('CPU')
    ax.set_zlabel('NETWORK IO')
    # plot train data
    ax.scatter(normal_features[:,-1:], normal_features[:,:1], normal_features[:,1:2], c='skyblue', marker='^')
    # plot test data according prediction
    if predictions is not None:
        for idx, pred in enumerate(predictions):
            curr_time = test_features[idx][2]
            curr_cpu = test_features[idx][0]
            curr_network = test_features[idx][1]
            ax.text(curr_time, curr_cpu + 1, curr_network + 1, (curr_cpu, curr_network), color=retrieveColor(pred))
            ax.scatter(curr_time, curr_cpu, curr_network, c=retrieveColor(pred), marker='o')
    if angle is not None:
        ax.view_init(30, 140 + angle)
        filename = 'static/plot' + str(int(angle/10 - 4)) + '.jpg'
    else:
        filename = 'static/plot' + str(idx) + '.jpg'
    plt.savefig(filename)

def plot_data_in_different_angle(features, test_features, prediction):
    for angle in range(5, 21):
        r = angle * 10
        plot_data(features, test_features, prediction, r)

def retrieveHourOfDay(h, size=1):
    x = np.arange(size, dtype=int)
    return np.full_like(x, 100 * h/ 24, dtype=np.double)

def generateTimeRange(startHour, endHour, batch_size):
    hoursofday = np.array([])
    for hour in range(startHour, endHour + 1):
        hoursofday = np.append(hoursofday, retrieveHourOfDay(hour, batch_size))
    return hoursofday

def generateCPULoadOrNetwork(start, end, data_size):
    return np.reshape(np.random.uniform(start, end, data_size), (data_size, 1))
