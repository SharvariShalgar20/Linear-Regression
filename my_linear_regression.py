import numpy as np

class Linear_Regression():

  #initializing the parameters
  def __init__(self, learning_rate, no_of_iteration):
    self.learning_rate = learning_rate
    self.no_of_iteration = no_of_iteration

  #X is year of experience(independent variable) and y is salary(dependent variable)
  def fit(self, X, y):
    #number of training examples & number of features
    self.m, self.n = X.shape
    #number of training(m) rows
    #number of feature(n colums)(1 because year of experience is the  feature and salary is target)
    #initiating weight and bias
    self.w = np.zeros(self.n)    #w will be decide on the number of features
    self.b = 0
    self.X = X
    self.y = y

    for i in range(self.no_of_iteration):
      self.update_weight()


  def update_weight(self):
    Y_prediction = self.predict(self.X)

    #calculate gradients

    dw = -(2 * (self.X.T).dot(self.y - Y_prediction)) / self.m

    db = -(2 * np.sum(self.y - Y_prediction))

    #updating the weights
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db


  def predict(self, X):
    return X.dot(self.w) +self.b
