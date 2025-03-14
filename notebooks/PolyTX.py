# Polynomial transformer Tool
# @arg: X: data matrix, order: target polynomial matrix order
# @rtnval: transformed and biased matrix P
# Ni Qingqing @ NUS, ECE, Created: AY24/25 S1

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv

def polyTx(X, order):
  poly = PolynomialFeatures(order)
  P = poly.fit_transform(X)

  return P

# Polynomial Regression Solving Tool
# @arg: P: polynomial matrix, y: label, 2d arr, ridge: True -> with ridge,
#       lamb: Ridge Param, ignored when ridge=False, Default = 0.01
# @rtnval: weight matrix
# Ni Qingqing @ NUS, ECE, Created: AY24/25 S1
# For rare cases that the equation (without ridge) is unsolvable with this tool,
# use solveLE() instead. With ridge, it's always solvable.

def solvePR(P, y, ridge=False, lamb=0.01):
  if ridge:
    if P.shape[0] > P.shape[1]:  # Primal
      w = inv(P.T @ P + lamb * np.identity(np.shape(P)[1])) @ P.T @ y
    else:                        # Dual
      w = P.T @ inv(P @ P.T + lamb * np.identity(np.shape(P)[0])) @ y

  else:
    if P.shape[0] > P.shape[1]:  # Primal
      w = inv(P.T @ P) @ P.T @ y
    else:                        # Dual
      w = P.T @ inv(P @ P.T) @ y

  return w


# Linear Regression with Ridge Solving Tool
# This tool calls solveRP, please keep them in the same file
# @arg: X: Biased dat matrix, y: label, 2d arr,
#       lamb: Ridge Param, Default = 0.01
# @rtnval: weight matrix
# Ni Qingqing @ NUS, ECE, Created: AY24/25 S1

def solveLE_Ridge(X, y, lamb=0.01):
  w = solvePR(X, y, ridge=True, lamb=lamb)
  return w