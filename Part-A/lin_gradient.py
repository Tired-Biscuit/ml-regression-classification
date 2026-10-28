import numpy as np
import matplotlib.pyplot as plt

x = [0.390561798,
0.339325843,
0.768988764,
0.683146067,
0.734831461,
0.617078652,
0.64,
0.328089888,
0.364044944,
1,
0.342022472,
0.883146067,
0.519550562,
0,
0.084494382,
0.554157303,
0.447191011,
0.106966292,
0.595505618,
0.174831461]

y = [0.609312067,
0.50153053,
0.128242307,
0.346060899,
0.285323022,
0.42274851,
0.58337361,
0.613823103,
0.706943773,
0,
0.657644595,
0.102464959,
0.432576124,
1,
0.870952151,
0.255356855,
0.48622523,
0.715965845,
0.378121476,
0.686483003]

X = np.column_stack((np.ones(len(x)), np.array(x)))
Y = np.array(y)

def hypothesis(X, theta):
  """Returns the hypothetic value"""
  return X @ theta

def compute_derived_cost_theta0(X, Y, theta):
  """Derivative of the SSE cost function with respect to the theta0 variable"""
  n = len(Y)
  return np.sum(hypothesis(X, theta) - Y)

def compute_derived_cost_theta1(X, Y, theta):
  """Derivative of the SSE cost function with respect to the theta1 variable"""
  n = len(Y)
  return np.sum((hypothesis(X, theta) - Y) * X[:, 1])

def gradient_descent(X, Y, init_theta, iterations, alpha=0.001):
  theta = np.array(init_theta)
  for i in range(iterations):
    d0 = theta[0] - alpha * compute_derived_cost_theta0(X, Y, theta)
    d1 = theta[1] - alpha * compute_derived_cost_theta1(X, Y, theta)

    theta = np.array((d0, d1))

  return theta

np.random.seed(42)
init_theta = np.random.randn(2, 1)

print("Initial random values for theta:", init_theta)

########### First iteration #############
theta = gradient_descent(X, Y, init_theta, 1)
H_1 = X @ theta

print("Final theta values after 1 iteration:", theta)

########### Second iteration #############
theta = gradient_descent(X, Y, init_theta, 2)
H_2 = X @ theta

print("Final theta values after 2 iterations:", theta)

########### 50 iterations #############
theta = gradient_descent(X, Y, init_theta, 50)
H_50 = X @ theta

print("Final theta values after 50 iterations:", theta)


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.scatter(np.array(x), Y, color="blue", label="Data points")
plt.plot(
    x,
    H_1,
    color="red",
    linestyle="-",
    label="linear regression (Least Squares)",
)
plt.title("Iteration 1")
plt.xlabel("Temperature")
plt.ylabel("Output")
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 2)
plt.scatter(np.array(x), Y, color="blue", label="Data points")
plt.plot(
    x,
    H_2,
    color="red",
    linestyle="-",
    label="linear regression (Least Squares)",
)
plt.title("Iteration 2")
plt.xlabel("Temperature")
plt.ylabel("Output")
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 3)
plt.scatter(np.array(x), Y, color="blue", label="Data points")
plt.plot(
    x,
    H_50,
    color="red",
    linestyle="-",
    label="linear regression (Least Squares)",
)
plt.title(f"Iteration 50")
plt.xlabel("Temperature")
plt.ylabel("Output")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
