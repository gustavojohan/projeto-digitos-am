import numpy as np

class LogisticRegression_:
    def __init__(self, eta=0.1, tmax=1000, bs=1000000):
      self.eta = eta
      self.tmax = tmax
      self.batch_size = bs

    # Infere o vetor w da funçao hipotese
    #Executa a minimizao do erro de entropia cruzada pelo algoritmo gradiente de descida
    def fit(self, _X, _y):
        X = np.array(_X)
        y = np.array(_y)

        N = X.shape[0]
        d = X.shape[1]

        self.w = np.zeros(d)

        for t in range(self.tmax):
            wx = X.dot(self.w)          # w^T x_i para todos os pontos
            ywx = y * wx                # y_i * w^T x_i
            v = y / (1 + np.exp(ywx))   # y_i / (1 + e^{y_i w^T x_i})
            gt = -(1/N) * X.T.dot(v)    # gradiente

            self.w = self.w - self.eta * gt
    
        
    #funcao hipotese inferida pela regressao logistica  
    def predict_prob(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    #Predicao por classificação linear
    def predict(self, X):
        prob = self.predict_prob(X)
        return np.where(prob >= 0.5, 1, 5)
    
    def getW(self):
        return self.w

    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]