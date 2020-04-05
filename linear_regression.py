import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def hypothesisFromModel(path, value):
    data = parseFile(path)
    print("The result for the value ", value, " is ", data[0] + data[1] * (value - data[2]) / data[3])

def costFunctionDerivatesComputation(theta, alpha, data):
    return np.array([theta[0] - alpha * np.sum(theta[0] + theta[1] * data[:, 0] - data[:, 1]) / data.size * 2, theta[1] - alpha * np.sum((theta[0] + theta[1] * data[:, 0] - data[:, 1]) * data[:, 0]) / data.size * 2])

def gradientDescent(theta, alpha, data, iters):
    for i in range(iters):
        theta[:] = costFunctionDerivatesComputation(theta, alpha, data)[:]
    return theta

def parseFile(path):
    try:
        return pd.read_csv(path).to_numpy().astype(float)
    except:
        print("The path does not correspond to a csv file.")
        sys.exit(0)

def plotData(data, theta, normalizer):
    x_axis = data[:, 0]
    y_axis = data[:, 1]
    plt.scatter(x_axis, y_axis, c="Blue")
    plt.plot([data.min(axis=0)[0], data.max(axis=0)[0]], [theta[0], theta[0] + theta[1] * (data.max(axis=0)[0] - normalizer[0]) / normalizer[1]], 'r')
    plt.xlabel("")
    plt.ylabel("Price (in euros)")
    plt.title("Linear regression for car price")
    plt.show()

def linearRegression(path):
    data = parseFile(path)
    data_plot = np.array(data[:]);
    theta = np.array([0.0, 0.0])
    alpha = 0.1
    normalizer = np.array([data.min(axis=0)[0], data.max(axis=0)[0] - data.min(axis=0)[0]]).astype(float)
    data[:, 0] = (data[:,0] - normalizer[0]) / normalizer[1]
    gradientDescent(theta, alpha, data, 10000)
    plotData(data_plot, theta, normalizer)
    pd.DataFrame(np.concatenate([theta, normalizer])).to_csv("ressources/theta.csv", index=False, sep=',')

linearRegression("ressources/data.csv")
hypothesisFromModel("ressources/theta.csv", 50000)
