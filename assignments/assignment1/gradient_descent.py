x_old = 0
x_new = 6
learning_rate = 0.01
precision = 0.00001

def f_derivative(x):
    return 4 * x**3 - 9 * x**2

while abs(x_new - x_old) > precision:
    x_old = x_new
    x_new = x_old - learning_rate * f_derivative(x_old)
    
print("Local minimun occure at ", x_new)