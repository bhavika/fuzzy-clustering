import matplotlib.pyplot as plt
import numpy as np


class InputSizeException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


def plot(gaussian_dist, mu, sigma):
    count1, bins1, ignored1 = plt.hist(gaussian_dist, 30, normed=True)
    plt.plot(bins1, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins1 - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')