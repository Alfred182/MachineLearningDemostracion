import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_random_data():
    rnstate = np.random.RandomState(1)
    x = 10 * rnstate.rand(50)
    y = 2 * x - 5 + rnstate.randn(50)
    return x[:, np.newaxis], y

def plot(x, y, modelo):
    xfit = np.linspace(0, 10, 1000)
    yfit = modelo.predict(xfit[:, np.newaxis])
    plt.plot(xfit, yfit)
    plt.scatter(x, y)
    plt.show()


x, y = get_random_data()
modelo = LinearRegression()
modelo.fit(x, y)
result = modelo.predict([[6]])
print(result)
plot(x, y, modelo)
