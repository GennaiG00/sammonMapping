import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

def readFromFile(fileName):
    with open(fileName, mode='r') as infile:
        reader = csv.reader(infile)
        data = [row for row in reader]
    return data

def writePointsToFile(fileName, data):
    with open(fileName, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        for point in data:
            writer.writerow(point)

def euclideanDistance(pointOne, pointTwo):
    if len(pointOne) == len(pointTwo):
        tot = 0
        for i in range(len(pointOne)):
            tot += np.abs(pointOne[i] - pointTwo[i]) ** 2
        return tot ** (1 / 2)

def sammonMapping(originalD, y, mf=0.3):
    n = y.shape[0]
    dimension = y.shape[1]
    updateD = np.array([[euclideanDistance(y[i], y[j]) for j in range(n)] for i in range(n)])
    yNew = np.copy(y)
    for p in range(n):
        for q in range(dimension):
            g = gradient(y, p, q, originalD, updateD)
            h = hessian(y, p, q, originalD, updateD)
            if h != 0:
                delta = g / abs(h)
            else:
                delta = 0
            yNew[p][q] -= mf * delta
    E = np.sum(((originalD - updateD) ** 2) * updateD) / c
    return yNew, E

def gradient(y, p, q, originalD, updateD):
    accum = 0
    for j in range(y.shape[0]):
        if j != p and originalD[p][j] > 0 and updateD[p][j] > 0:
            a = (originalD[p][j] - updateD[p][j]) / (originalD[p][j] * updateD[p][j])
            b = y[p][q] - y[j][q]
            accum += a * b
    return (-2 / c) * accum

def hessian(y, p, q, originalD, updateD):
    accum = 0
    for j in range(y.shape[0]):
        if j != p and updateD[p][j] > 0 and originalD[p][j] > 0:
            a = 1 / (originalD[p][j] * updateD[p][j])
            b = originalD[p][j] - updateD[p][j]
            e = ((y[p][q] - y[j][q]) ** 2) / updateD[p][j]
            d = 1 + ((originalD[p][j] - updateD[p][j]) / updateD[p][j])
            accum += a * (b - e * d)
    return (-2 / c) * accum

def plot_data(y, labels):
    plt.figure(figsize=(8, 6))
    uniqueLabels = np.unique(labels)
    if dimension == 2:
        for label in uniqueLabels:
            idx = labels == label
            plt.scatter(y[idx, 0], y[idx, 1], label=f"Class {int(label)}", s=50)
        plt.xlabel("X")
        plt.ylabel("Y")
    elif dimension == 3:
        ax = plt.axes(projection='3d')
        for label in uniqueLabels:
            idx = labels == label
            ax.scatter(y[idx, 0], y[idx, 1], y[idx, 2], label=f"Class {int(label)}", s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.title("Sammon Mapping Results")
    plt.legend()
    plt.show()


def plotError(E, interval=5):
    plt.figure(figsize=(8, 6))
    if len(E) > 100:
        interval = 10
    plt.plot(E, marker='o')
    plt.xticks(np.arange(0, len(E), interval))
    plt.title("Error E over Iterations")
    plt.xlabel(f"Iteration (every {interval})")
    plt.ylabel("Error E")
    plt.grid()
    plt.show()



if __name__ == '__main__':
    if len(sys.argv) < 2:
        datasetName = "blobs"
    else:
        datasetName = sys.argv[1]
    dataPos = readFromFile('../' + datasetName + '.csv')
    dataLabels = readFromFile('../' + datasetName + '_labels.csv')
    dataFloat = np.array([[float(value) for value in row] for row in dataPos])
    labels = np.array([float(label[0]) for label in dataLabels])
    originalD = np.zeros((dataFloat.shape[0], dataFloat.shape[0]))

    for i in range(dataFloat.shape[0]):
        for j in range(i + 1, dataFloat.shape[0]):
            dist = euclideanDistance(dataFloat[i], dataFloat[j])
            originalD[i][j] = dist
            originalD[j][i] = dist

    iter = 20
    dimension = 2
    mf = 0.3
    initialization = "random"
    if len(sys.argv) == 3:
        iter = int(sys.argv[2])
    elif len(sys.argv) == 4:
        iter = int(sys.argv[2])
        mf = float(sys.argv[3])
    elif len(sys.argv) == 5:
        initialization = sys.argv[4]
        iter = int(sys.argv[2])
        mf = float(sys.argv[3])
    elif len(sys.argv) == 6:
        dimension = int(sys.argv[5])
        initialization = sys.argv[4]
        iter = int(sys.argv[2])
        mf = float(sys.argv[3])

    rnd = np.random.default_rng()
    if initialization == "random":
        y = rnd.random((len(originalD), dimension))
    elif initialization == "orthogonal":
        cov_matrix = np.cov(dataFloat, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigvals)[::-1]
        principal_components = eigvecs[:, sorted_indices[:dimension]]
        y = np.dot(dataFloat, principal_components)
    c = np.sum(1 / (originalD[originalD != 0]))

    E = []
    for itr in range(iter):
        yNew, e = sammonMapping(originalD, y, mf=mf)
        E.append(e)
        y = np.copy(yNew)

    writePointsToFile(datasetName + "Result.csv", y)
    plot_data(y, labels)
    plotError(E)
