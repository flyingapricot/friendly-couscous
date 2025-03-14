# LE Solving Tool
# Ni Qingqing @ NUS, ECE, Created: AY22/23 S2
# Ni Qingqing @ NUS, ECE, Updated: AY24/25 S1
'''
Log
1. det() -> det_checker(), avoid shadowing of np.linalg.det()
2. solveLE() now returns w
'''

import numpy as np
from numpy.linalg import inv, matrix_rank,det

'''Helper functions'''

def det_checker(X):
  """
  Determines whether a matrix is square, overdetermined, or underdetermined.

  Parameters:
  X (numpy.ndarray): A 2D NumPy array (matrix).

  Returns:
  str: 
      - "even" if the matrix is square (rows == columns).
      - "over" if the matrix is overdetermined (more rows than columns).
      - "under" if the matrix is underdetermined (more columns than rows).
  """
  m = np.shape(X)[0]
  d = np.shape(X)[1]
  if m==d:
    detX = "even"
  elif m>d:
    detX = "over"
  else:
    detX = "under"

  return detX

def RC_checker(X,y):
  X_ = np.append(X, y ,axis=1)
  rankX = matrix_rank(X)
  rankX_ = matrix_rank(X_)
  d = np.shape(X)[1]

  if rankX == rankX_:
    if rankX == d:
      RC = 1
    else:
      RC = 3
  else:
    RC = 2
  return RC, rankX, rankX_

def evenSolver(X,y):
  w = None
  RC, _,_ = RC_checker(X,y)
  if RC==1:
    w = inv(X) @ y
    ans = "unique."
  elif RC==2:
    ans = "No solution."
  else:
    ans = "Infinitely many solutions."
  return w, ans

def overSolver(X,y):
  w = None
  RC, _,_ = RC_checker(X,y)
  if RC==1:
    w = inv(X.T @ X) @ X.T @ y
    ans = "unique."
  elif RC==3:
    ans = "Infinitely many solutions."
  elif det_checker(X.T @ X) != 0:
    w = inv(X.T @ X) @ X.T @ y
    ans = "No exact solution, but least square approximation can be found."
  else:
    ans = "No solution."
  return w, ans

def underSolver(X,y):
  w = None
  RC, _,_ = RC_checker(X,y)
  if RC==2:
    ans = "No solution."
  elif det_checker(X @ X.T) != 0:
    w = X.T @ inv(X @ X.T) @ y
    ans = "No exact solution, but least norm approximation can be found."
  else:
    ans = "Infinitely many solutions."
  return w, ans

def solveLE(X,y):
  """
  Solves a system of linear equations (LE) based on its nature (even, overdetermined, or underdetermined).

  The function determines whether the system is:
  - **Square ("even")**: Uses `evenSolver(X, y)`.
  - **Overdetermined ("over")**: Uses `overSolver(X, y)`.
  - **Underdetermined ("under")**: Uses `underSolver(X, y)`.

  Parameters:
  -----------
  X : numpy.ndarray
      The coefficient matrix of shape (m, d), where `m` is the number of equations and `d` is the number of variables.
  y : numpy.ndarray
      The right-hand side (output) vector of shape (m, 1).

  Returns:
  --------
  w : numpy.ndarray
      The computed solution vector.
  
  Prints:
  -------
  - The nature of the system (even, overdetermined, underdetermined).
  - The rank information and solution method used.
  - The computed solution `w`.

  Usage:
  ------
  ```python
  from LESolver import *
  import numpy as np

  # Define system of equations
  X = np.array([[1, 3], [1, 4], [1, 5], [1, 6], [1, 7]])
  y = np.array([[5], [4], [3], [2], [1]])

  # Determine nature of LE system
  print("The LE system is", det_checker(X), "- determined.\n")

  # Check rank and solve
  cases, rankX, rankX_ = RC_checker(X, y)
  print("rank(X) =", rankX, "\nrank(X~) =", rankX_, "\nfalls into case", cases)

  # Solve the system
  solution = solveLE(X, y)
  ```

  Notes:
  ------
  - This function assumes that `evenSolver`, `overSolver`, and `underSolver` are defined in `LESolver`.
  - Ensure `LESolver` is correctly imported before calling `solveLE`.
  -	Sometimes the linear equations are not given in the standard form as Xw = y. 
	- Transformation is required before applying the rules. 
  - If A and B are both matrices, (AB)^T=B^T A^T
  """
  detX = det_checker(X)
  if detX == "even":
    w, ans = evenSolver(X,y)
  elif detX == "over":
    w, ans = overSolver(X,y)
  else:
    w, ans = underSolver(X,y)

  print("\n", ans, "\nw =", w)

  return w


#From YuWen
def check_invertibility(X):
    '''
    X is a (m,d) matrix. m is samples, d is features
    outputs true false values for (left_inverse_exist,right_inverse_exists)
    doesnt check exactly equal to 0 as det might be vv small
    '''
    use_LI = True if (det(X.T @X) - 0.001 > 0) else False
    use_RI = True if (det(X @ X.T) - 0.001 > 0) else False
    return (use_LI,use_RI)
