import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, x, y):
        self.weights = np.zeros(1 + x.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0

            for input, target in zip(x, y):
                y_pred = self.predict(input)
                update = self.lr*(target - y_pred)
                self.weights[0] += update
                self.weights[1:] += update*input
                errors += int(update != 0)
            
            self.errors.append(errors)

        return self
    
    def net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, -1)
    
    
# Uncomment the following lines to test the model

# train_x = np.array([[2.0, 1.0], [3.0, 4.0], [4.0, 2.0], [3.0, 1.0]])
# train_y = np.array([-1, 1, 1, -1])
# test_x = np.array([[5.0, 2.0], [1.0, 3.0]])

# model = Perceptron()
# model.fit(train_x, train_y)
# print(model.predict(test_x))
