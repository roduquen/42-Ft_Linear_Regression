import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hypothesisReturn(theta0, theta1, value):
    result = theta0 + theta1 * value
    return result

def costFunctionDerivatesComputation(mileage, price, data_nbr, theta0, theta1, alpha):
    derivate0, derivate1 = 0.0, 0.0
    for i in range(0, data_nbr):
        tmp = theta0 + theta1 * mileage[0, i] - price[0, i]
        derivate0 += tmp
        derivate1 += tmp * mileage[0, i]
    return theta0 - alpha * (derivate0 / data_nbr), theta1 - alpha * (derivate1 / data_nbr)

def normalizeValue(value, min, divider):
    result = float(value - min) / divider
    return result

def normalizeVector(vector, size, min, divider):
    vector_new = vector[0, 0:size]
    for i in range(0, size):
        vector_new[0, i] = normalizeValue(vector_new[0, i], min, divider)
    return vector_new

def gradientDescent(mileage, price, data_nbr, theta0, theta1, alpha, min, divider):
    prev_theta0, prev_theta1 = theta0 + 1, theta1 + 1
    mileage_normalized = normalizeVector(mileage, data_nbr, min, divider)
    epsilon = 1e-20
    while abs(prev_theta0 - theta0) > epsilon or abs(prev_theta1 - theta1) > epsilon:
        tmp0, tmp1 = costFunctionDerivatesComputation(mileage_normalized, price, data_nbr, theta0, theta1, alpha)
        prev_theta0, prev_theta1 = theta0, theta1
        theta0, theta1 = tmp0, tmp1
    return theta0, theta1

def convertFileToMatrix(file):
    try:
        data_nbr = file.shape[0]
        mileage = np.matrix(file.iloc[0:data_nbr, 0].values).astype(float)
        price = np.matrix(file.iloc[0:data_nbr, 1].values).astype(float)
    except:
        print("Fail to create the matrices")
        sys.exit(0)
    return mileage, price, data_nbr

def parseFile(path):
    try:
        file = pd.read_csv(path)
    except:
        print("The path does not correspond to a csv file.")
        sys.exit(0)
    return file

def main():
    file = parseFile("ressources/data.csv")
    mileage, price, data_nbr = convertFileToMatrix(file)
    theta0, theta1 = 0.0, 0.0
    alpha = 0.1
    min = mileage.min()
    divider = float(mileage.max() - min)
    theta0, theta1 = gradientDescent(mileage, price, data_nbr, theta0, theta1, alpha, min, divider)
    print("The local optimal for theta0 is ", theta0, " and for theta1 is ", theta1)
    value = 50000
    value_normalized = normalizeValue(value, min, divider)
    print("For the value ", value, ", the result is : ", hypothesisReturn(theta0, theta1, value_normalized))

main()

