# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 2016

@author: jphong
"""
import classificationMethod
import numpy as np
import util

class LogisticRegressionClassifier(classificationMethod.ClassificationMethod):
  def __init__(self, legalLabels, type, seed):
    self.legalLabels = legalLabels
    self.type = type
    self.learningRate = [0.01, 0.001, 0.0001]
    self.l2Regularize = [1.0, 0.1, 0.0]
    self.numpRng = np.random.RandomState(seed)
    self.initialWeightBound = None
    self.posteriors = []
    self.costs = []
    self.epoch = 1000
    self.bestParam = None # You must fill in this variable in validateWeight

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method.
    Iterates several learning rates and regularization parameter to select the best parameters.

    Do not modify this method.
    """
    for lRate in self.learningRate:
      curCosts = []
      for l2Reg in self.l2Regularize:
        self.initializeWeight(trainingData.shape[1], len(self.legalLabels))
        for i in xrange(self.epoch):
          cost, grad = self.calculateCostAndGradient(trainingData, trainingLabels)
          self.updateWeight(grad, lRate, l2Reg)
          curCosts.append(cost)
        self.validateWeight(validationData, validationLabels)
        self.costs.append(curCosts)

  def initializeWeight(self, featureCount, labelCount):
    """
    Initialize weights and bias with randomness.

    Do not modify this method.
    """
    if self.initialWeightBound is None:
      initBound = 1.0
    else:
      initBound = self.initialWeightBound
    self.W = self.numpRng.uniform(-initBound, initBound, (featureCount, labelCount))
    self.b = self.numpRng.uniform(-initBound, initBound, (labelCount, ))

  def calculateCostAndGradient(self, trainingData, trainingLabels):
    """
    Fill in this function!

    trainingData : (N x D)-sized numpy array
    trainingLabels : N-sized list
    - N : the number of training instances
    - D : the number of features (PCA was used for feature extraction)
    RETURN : (cost, grad) python tuple
    - cost: python float, negative log likelihood of training data
    - grad: gradient which will be used to update weights and bias (in updateWeight)

    Evaluate the negative log likelihood and its gradient based on training data.
    Gradient evaluted here will be used on updateWeight method.
    Note the type of weight matrix and bias vector:
    self.W : (D x C)-sized numpy array
    self.b : C-sized numpy array
    - D : the number of features (PCA was used for feature extraction)
    - C : the number of legal labels
    """

    "*** YOUR CODE HERE ***"
    """
     Note
     ----
          NLL  = tr(X W Y) + N_1^T log(exp(X W) C_1)

          grad(w) = (M-Y)X (+ \lambda W should be done at update)
          grad(b) = (M-Y)N_1
    """
    X = np.array(trainingData)
    W = np.array(self.W)
    Y = np.stack([[1 if i==trainingLabels[j] else 0 for i in range(len(self.legalLabels))] for j in range(len(trainingLabels))]).T
        # trainingLabels Matrix with (C x N)-sized
    N_1 = np.ones(len(trainingLabels))
        # one vector with (N)-sized
    C_1 = np.ones(len(self.legalLabels))
        # one vector with (C)-sized
    B = np.outer(N_1, np.array(self.b))
        # b Matrix with (N x C)-sized
    M = np.stack([util.normalize(row) for row in (np.exp(np.dot(X,W) + B))]).T
        # \mu Matrix with (C x N)-sized

    cost = -np.trace(np.dot(np.dot(X,W),Y)) + np.dot(N_1.T, np.log(np.dot(np.exp(np.dot(X,W)),C_1)))
    grad = {'w': np.dot(M-Y,X).T, 'b': np.dot(M-Y,N_1)}

    return cost, grad

  def updateWeight(self, grad, learningRate, l2Reg):
    """
    Fill in this function!
    grad : gradient which was evaluated in calculateCostAndGradient
    learningRate : python float, learning rate for gradient descent
    l2Reg: python float, L2 regularization parameter

    Update the logistic regression parameters using gradient descent.
    Update must include L2 regularization.
    Please note that bias parameter must not be regularized.
    """

    "*** YOUR CODE HERE ***"

    self.W -= learningRate * (grad['w'] + l2Reg * self.W)
    self.b -= learningRate * grad['b']

  def validateWeight(self, validationData, validationLabels):
    """
    Fill in this function!

    validationData : (M x D)-sized numpy array
    validationLabels : M-sized list
    - M : the number of validation instances
    - D : the number of features (PCA was used for feature extraction)

    Choose the best parameters of logistic regression.
    Calculates the accuracy of the validation set to select the best parameters.
    """

    "*** YOUR CODE HERE ***"

    if self.bestParam is None:
      self.accuracy = -99999
      self.bestParam = (self.W, self.b)  # initialize accuracy
    guesses = []
    for datum in validationData:
      logposterior = util.normalize(np.exp(np.dot(self.W.T, datum) + self.b))
      guesses.append(np.argmax(logposterior))
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    accuracy = 100.0 * correct / len(validationLabels)
    print("accuracy : " + str(accuracy) + " %")
    if self.accuracy < accuracy:
      self.accuracy = accuracy
      self.bestParam = (self.W, self.b)


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    Do not modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      logposterior = self.calculateConditionalProbability(datum)
      guesses.append(np.argmax(logposterior))
      self.posteriors.append(logposterior)

    return guesses

  def calculateConditionalProbability(self, datum):
    """
    datum : D-sized numpy array
    - D : the number of features (PCA was used for feature extraction)
    RETURN : C-sized numpy array
    - C : the number of legal labels

    Returns the conditional probability p(y|x) to predict labels for the datum.
    Return value is NOT the log of probability, which means
    sum of your calculation should be 1. (sum_y p(y|x) = 1)
    """

    bestW, bestb = self.bestParam # These are parameters used for calculating conditional probabilities

    "*** YOUR CODE HERE ***"
    prob = util.normalize(np.exp(np.dot(bestW.T, datum) + bestb))

    return prob
