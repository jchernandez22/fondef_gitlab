import numpy as np
from sklearn.neural_network import MLPRegressor
from services.load_dataset import load_dataset

class NNModel():
    def __init__(self, layers, activation, train=False):
        self.model = MLPRegressor(hidden_layer_sizes = layers, 
                        activation = activation, 
                        solver = 'adam', 
                        learning_rate_init = 0.01, 
                        learning_rate='adaptive',
                        max_iter = 1000000, 
                        tol = 0.000001, 
                        random_state=1,
                        verbose = False)
        
        self.X, self.y = load_dataset(split=True)

        if train:
            self.fit_model()
    
    def fit_model(self,):
        self.model.fit(self.X, self.y)
    
    def predict(self, X):
        return self.model.predict(X)

if __name__ == '__main__':
    NNModel(activation="relu",
                       layers=[1,1,1],
                       train=False)