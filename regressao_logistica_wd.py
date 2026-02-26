import numpy as np

class LogisticRegressionWD_:
    def __init__(self, eta=0.1, tmax=1000, lamb=0.01):
        self.eta = eta
        self.tmax = tmax
        self.lamb = lamb # Parâmetro de regularização (lambda)

    def fit(self, _X, _y):
        X = np.array(_X)
        y = np.array(_y)
        N, d = X.shape
        self.w = np.zeros(d)

        for t in range(self.tmax):
            wx = X.dot(self.w)
            v = y / (1 + np.exp(y * wx))
            
            # Gradiente original + termo de regularização (Weight Decay)
            gt = -(1/N) * X.T.dot(v) + (2 * self.lamb / N) * self.w
            
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