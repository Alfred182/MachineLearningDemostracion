import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_random_data():
    x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
    y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
    return x,y

def plot(x, y, modelo):
    line = np.linspace(0, 30, 100)
    plt.plot(line, modelo(line))
    plt.scatter(x, y)
    plt.show()


x, y = get_random_data()
modelo = np.poly1d(np.polyfit(x, y, 3))
plot(x, y, modelo)
