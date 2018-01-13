import random
import numpy as np
import pandas as pd
import math
from sklearn.metrics import homogeneity_completeness_v_measure
from utilities import *

MAX_ITER = 500
epsilon = 0.01


def create_data(mu, sigma, n, p, random_seed):
    """
    Generate data using the given parameters
    :param mu: list of means
    :param sigma: list of standard deviations
    :param n: number of data points 
    :param p: probabilities of drawing from each distribution
    :param random_seed: seed for initialization
    :return: pandas dataframe with data 'x' and corresponding label
    """
    random.seed(random_seed)
    x = []
    y_true = []

    if len(mu) != len(sigma) != len(p):
        raise InputSizeException("The following iterables need to be of equal length {} {}".format(mu, sigma, p))
    else:
        n_distributions = len(mu)

    for i in range(n_distributions):
        size = int(n*p[i])
        gauss = np.random.normal(loc=mu[i], scale=sigma[i], size=size).T
        x.extend(gauss)
        y = [i] * size
        y_true += y

    data = {'x': x, 'label': y_true}
    df = pd.DataFrame(data=data)

    return df


def FCM(dataframe, random_state=0, C=3, m=2):
    """
    Fuzzy C-means Clustering. 
    :param dataframe: 
    :param random_state: 
    :param c: int, no. of clusters 
    :param m: int, fuzzifier: the level of cluster fuzziness
    :return: list of predicted labels, list of cluster centroids 
    """
    random.seed(random_state)
    D = dataframe.shape[0]

    data = np.array(dataframe['x'])
    labels = np.array(dataframe['label'])

    n_iter = 0

    # W_ij is the value in matrix W which tells us how likely it is that value i in data belongs to cluster j
    # i = 500, j = 3 (no. of clusters).
    # Therefore our W matrix is of size (i rows * j columns)

    W = list()
    for i in range(D):
        c = np.random.dirichlet(np.ones(3) / 3, size=1)
        c = c.tolist()
        W.append(c)

    centroids = calculate_centroids(data, W, C, D, m)
    j_m = Jm(data, W, C, D, centroids, m)
    j_m2 = 0

    while not stopping_criterion(n_iter, j_m, j_m2):
        j_m2 = j_m
        centroids = calculate_centroids(data, W, C, D, m)
        W = update_membership(W, centroids, C, data, m)
        labels = getClusters(W, D)
        n_iter += 1

    return labels, centroids


def Jm(data, W, C, D, centroids, m):
    """
    Objective function calculation 
    :param data: numpy array
    :param W: list of lists
    :param C: int, no of clusters
    :param D: int, no. of datapoints
    :param centroids: list, centroids 
    :param m: int, fuzzifier
    :return: int, value of objective function
    """
    mu = 0
    for i in range(D):
        for j in range(C):
            dist = [data[i] - centroids[c] for c in range(C)]
            mu += (float(W[i][0][j]) ** m) * (sum(dist) ** 2)
    return mu


def getClusters(W, D):
    """
    For each data point, get original labels
    :param W: list of lists 
    :param D: int, size of dataset
    :return: list, cluster labels
    """
    cluster_labels = list()
    for i in range(D):
        temp = (W[i][0])
        idx = temp.index(min(temp))
        cluster_labels.append(idx)
    return cluster_labels


def update_membership(W, centroids, C, data, m):
    p = float(2/(m-1))
    for i in range(len(data)):
        x = data[i]
        numerator = [x - centroids[j] for j in range(C)]
        for j in range(C):
            denominator = sum([math.pow(float(numerator[j]/numerator[c]), p) for c in range(C)])
            W[i][0][j] = float(1/denominator)
    return W


def stopping_criterion(n_iter, j_m, j_m2):
    """
    Stopping criterion in fuzzy clustering, either:
        1. n_iter > MAX_ITER
        2. |j_m - j_m2| < epsilon
    :param n_iter: int, current iteration
    :param j_m: int
    :param j_m2: int 
    :return: 
    """
    if n_iter > MAX_ITER:
        return True
    elif abs(j_m - j_m2) < epsilon:
        return True
    else:
        return False


def calculate_centroids(data, W, C, D, m):
    centroids = []

    for j in range(C):
        x = []
        for i in range(D):
            x.append(W[i][0][j])
        xraised = [x_i ** m for x_i in x]
        denominator = sum(xraised)
        temp = zip(xraised, data)
        numerator = 0
        for i in temp:
            numerator += i[0] * float(i[1])
        c_j = numerator/denominator
        centroids.append(c_j)
    return centroids
