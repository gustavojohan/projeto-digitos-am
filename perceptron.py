import matplotlib.pyplot as plt
import numpy as np
import random
from IPython.display import clear_output

def PLA(X, y):
    """
    Esta função corresponde ao Algoritmo de Aprendizagem do modelo Perceptron.
    
    Paramêtros:
    - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
    às coordenadas dos pontos gerados.
    - y (list): Classificação dos pontos da amostra X.
    - f (list): Lista de dois elementos correspondendo, respectivamente, aos coeficientes angular e linear 
    da função alvo.
    
    Retorno:
    - it (int): Quantidade de iterações necessárias para corrigir todos os pontos classificados incorretamente.
    - w (list): Lista de três elementos correspondendo aos pesos do perceptron.
    """
    w = np.array([0.0, 0.0, 0.0])
    listaPCI, new_y = constroiListaPCI(X, y, w)
    it = 0
    
    while (len(listaPCI) > 0):
        # Pega um indice aleatorio
        i = random.choice(range(len(listaPCI)))

        w = w + new_y[i] * listaPCI[i]
        it += 1
        # Após atualizar os pesos para correção do ponto escolhido, você irá chamar a função plotGrafico()
        plot_grafico(X, y, w)
        # Aqui você deverá construir a lista de pontos classificados incorretamente
        listaPCI, new_y = constroiListaPCI(X, y, w)
        
    return it, w

def constroiListaPCI(X, y, w):
    """
    Esta função constrói a lista de pontos classificados incorretamente.
    
    Paramêtros:
    - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
    às coordenadas dos pontos gerados.
    - y (list): Classificação dos pontos da amostra X.
    - w (list): Lista correspondendo aos pesos do perceptron.
   
    Retorno:
    - l (list): Lista com os pontos classificados incorretamente.
    - new_y (list): Nova classificação de tais pontos.
   
    """

    l = []
    new_y = []

    for i in range(len(X)):
        h = np.sign(X[i] @ w)
        if h == 0:
            h = 1
        if h != y[i]:
            l.append(X[i])
            new_y.append(y[i])
            
    
    
    return l, new_y

def calcular_acuracia(X_teste, y_teste, w):
    # Obtém previsões do perceptron
    y_pred = np.sign(X_teste @ w)

    # Ajusta valores 0 para +1
    y_pred[y_pred == 0] = 1
    
    # Compara com os rótulos reais, retorna a quantidade de acertos
    acertos = np.sum(y_pred == y_teste)
    
    # Calcula a acurácia percentual
    acuracia = (acertos / len(y_teste)) * 100
    return acuracia

def plot_grafico(X, y, w):   
    plt.clf()

    # pontos
    positivos = X[y == 1]
    negativos = X[y == -1]

    plt.scatter(positivos[:, 1], positivos[:, 2], s=10, c="blue", label="1")
    plt.scatter(negativos[:, 1], negativos[:, 2], s=10, c="red", label="5")

    # reta aprendida: w0 + w1*x + w2*y = 0 => y = (-w0 - w1*x)/w2
    x_line = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    if w[2] != 0:
        y_line = (-w[0] - w[1]*x_line) / w[2]
        plt.plot(x_line, y_line, color="green")

    plt.xlabel("Intensidade")
    plt.ylabel("Simetria")
    plt.legend()
    plt.grid(True)
    plt.xlim(min(X[:, 1]) - 0.1, max(X[:, 1]) + 0.1)
    plt.ylim(min(X[:, 2]) - 0.1, max(X[:, 2]) + 0.1)
    clear_output(wait=True)
    plt.show()