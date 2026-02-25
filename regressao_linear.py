import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def fit(self, _X, _y):
        _X = np.array(_X)
        _y = np.array(_y)

        self.w = np.linalg.pinv(_X) @ _y
     
    def predict(self, _x):
        return _x @ self.w
     
    def getW(self):
        return self.w
    
def plot_regressao_linear(X, y, model):
    plt.clf()

    # pontos
    positivos = X[y == 1]
    negativos = X[y == -1]

    plt.scatter(positivos[:, 1], positivos[:, 2], s=10, c="blue", label="1")
    plt.scatter(negativos[:, 1], negativos[:, 2], s=10, c="red", label="5")

    # reta aprendida: w0 + w1*x + w2*y = 0 => y = (-w0 - w1*x)/w2
    w = model.getW()
    x_line = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    if w[2] != 0:
        y_line = (-w[0] - w[1]*x_line) / w[2]
        plt.plot(x_line, y_line, color="green")

    plt.xlabel("Intensidade")
    plt.ylabel("Simetria")
    plt.legend()
    plt.grid(True)
    plt.show()