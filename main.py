import numpy as np
import pandas as pd
import math, copy
from matplotlib import pyplot as plt
from lab_utils_common import dlblue, dlorange, dldarkred, dlmagenta, dlpurple, dlcolors

from scipy import stats
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl, plt_gradients
plt.style.use('./deeplearning.mplstyle')

data = pd.read_csv('bottle.csv', delimiter=',', low_memory=False)


data =  data[data[['T_degC', 'Salnty']].notnull().all(1)]
data = np.array(data)

data = data[: ,[5,6]]
data_train = data[0:1000].T

x_train = data_train[0]
y_train = data_train[1]
cost_history=[]

def cost_function(w,b,x,y):
    m = 1000
    j =0

    for i in range(m):
        j_i = (w*x[i] + b - y[i])**2
        cost_history.append(j_i)
        j+=j_i

    return j/(2*m)
w =-0.15
b =34.9

def get_gradient(m,x,y,w,b):
    djwb_dw = 0
    djwb_db = 0
    for i in range(m):
        f_wb_i = w*x[i]+b
        djwb_dw += (f_wb_i - y[i])*x[i]
        djwb_db += (f_wb_i - y[i])

    return djwb_dw/m, djwb_db/m


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      y (ndarray (m,)): target values
    """
    m = 1000
    f_wb = np.zeros(m)
    print(f_wb)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        print(f_wb[i])
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)
print(tmp_f_wb)
J = cost_function(w,b,x_train,y_train)




def gradient_descent(alpha, w_in,b_in,x,y, iterations, cost_fn, gradient_function):
    w = w_in
    b = b_in
    m = 1000
    J_history = []
    p_history = []

    for i in range(iterations):

        djwb_dw, djwb_db = gradient_function(m, x, y,w,b)
        w = w_in - alpha*djwb_dw
        b = b_in - alpha*djwb_db
        w_in = w
        b_in = b
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(w,b,x,y))
            p_history.append([w, b])
        if i% math.ceil(iterations/10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]} ",f"dj_dw: {djwb_dw}, dj_db: {djwb_db}  ",f"w: {w}, b:{b}")

    return w, b, J_history, p_history #return w and J,w history for graphing


w_in=0
b_in=0
alpha = 0.01
iterations =10000


w_final, b_final, J_hist, p_hist = gradient_descent(alpha, w_in,b_in,x_train,y_train, iterations, cost_function, get_gradient)

print(f"(w,b) found by gradient descent: ({w_final},{b_final})")

plt.subplot(1,2,1)
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
plt.plot()

plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.grid()


w_range = np.array([-75, 75])
b_range = np.array([-100, 100])
b_space  = np.linspace(*b_range, 100)
w_space  = np.linspace(*w_range, 100)

tmp_b,tmp_w = np.meshgrid(b_space,w_space)
z=np.zeros_like(tmp_b)
for i in range(tmp_w.shape[0]):
    for j in range(tmp_w.shape[1]):
        z[i,j] = cost_function(tmp_w[i][j], tmp_b[i][j],x_train, y_train)
        if z[i,j] == 0: z[i,j] = 1e-6

plt.subplot(1,2,2)
plt.contour(tmp_w, tmp_b, np.log(z),levels=12, linewidths=2, alpha=0.7,colors=dlcolors)
plt.grid()
plt.show()