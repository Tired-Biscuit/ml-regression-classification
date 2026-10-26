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

theta = np.linalg.inv(X.T @ X) @ X.T @ Y

print(f"Theta 0: {theta[0]}, theta 1: {theta[1]}")

H = X @ theta

print(H)
plt.scatter(np.array(x), Y, color="blue", label="Data points")
plt.plot(x, H, color="red", linestyle="-", label="linear regression (Least Squares)")
plt.title(f"Using Least Squares (formal solution)")
plt.xlabel("Temperature")
plt.ylabel("Output")
plt.legend()
plt.show()
