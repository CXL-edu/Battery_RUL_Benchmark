import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.animation as animation

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
sklearn.utils.check_random_state(random_seed)


def build_model(model_name: str):
    model_dict = {
        'LR': 'from sklearn.linear_model import LinearRegression as Model',
        'KNN': 'from sklearn.neighbors import KNeighborsRegressor as Model',
        'SVR': 'from sklearn.svm import SVR as Model',
        'GPR': 'from sklearn.gaussian_process import GaussianProcessRegressor as Model',
        'MLP': 'from sklearn.neural_network import MLPRegressor as Model',
        'RF': 'from sklearn.ensemble import RandomForestRegressor as Model',
        'LightGBM': 'from lightgbm import LGBMRegressor as Model',
    }
    exec(model_dict[model_name])
    _model = eval("Model()")
    return _model

def build_criterion(criterion_name: str):
    crteria_dict = {
        'MSE': 'from sklearn.metrics import mean_squared_error as criterion',
        'RMSE': 'from sklearn.metrics import mean_squared_error as criterion',   # squared=False
        'MAE': 'from sklearn.metrics import mean_absolute_error as criterion',
    }

    exec(crteria_dict[criterion_name])
    _criterion = eval("criterion")
    # _criterion = eval("criterion()") if criterion_name not in ['RMSE'] else eval("criterion(squared=False)")
    return _criterion


seq_len, pred_len = 10, 8   # Time step of input and output
input_dim, output_dim = 1, 1 # Feature number of input and output
data_source = 'NASA' # 'NASA' or 'CMAPSS'
standard_flag = True # True or False

model_name = 'LR' # 'LR' or 'GPR'
criterion = 'RMSE' # 'MSE' or 'RMSE' or 'MAE'


data_path = {
    'NASA': 'data/NASA/',
}
data_list_npy = {
    'NASA': 'NASA',
}
data_list_csv = {
    'NASA': ['B0005', 'B0006', 'B0007', 'B0018'],
}


# Load data from csv-type file
data = []
for file_name in data_list_csv[data_source]:
    data.append(pd.read_csv(data_path[data_source]+file_name+'.csv').to_numpy())
    print('loading data: ', file_name+'.csv')

# # Load data from npy-type file
# data, data_temp = [], np.load(data_path[data_source]+data_list_npy[data_source]+'.npy', allow_pickle=True).item()
# for file_name in data_list_csv[data_source]:
#     data.append(np.array(data_temp[file_name]).transpose())
#     print('loading data: ', file_name+'.npy')


# Process data to RUL task
def process_data2RUL(data, seq_len, pred_len):
    data_x, data_y = [], []
    if seq_len == pred_len:
        for i in range(len(data)):
            data_temp = []
            for j in range(len(data[i])-2*seq_len+1):
                data_temp.append(data[i][j:j+seq_len, -1])
            data_x.append(data_temp[:-seq_len])
            data_y.append(data_temp[seq_len:])
            print(len(data_x[-1]), len(data_y[-1]))
    else:
        for i in range(len(data)):
            data_temp_x, data_temp_y = [], []
            for j in range(len(data[i])-seq_len-pred_len+1):
                data_temp_x.append(data[i][j:j+seq_len, -1])
                data_temp_y.append(data[i][j+seq_len:j+seq_len+pred_len, -1])
            data_x.append(data_temp_x)
            data_y.append(data_temp_y)
            print(len(data_x[-1]), len(data_y[-1]))

    return data_x, data_y

data_x, data_y = process_data2RUL(data, seq_len, pred_len)


# Split data into train and test set    # ['B0005.csv', 'B0006.csv', 'B0018.csv', 'B0007.csv']
# train_x, train_y = [np.array(data_x[i]) for i in [0,1,3]], [np.array(data_y[i]) for i in [0,1,3]]
# test_x, test_y = np.array(data_x[2]), np.array(data_y[2])
# real_capacity = data[2][:, -1]

train_x, train_y = [np.array(data_x[i]) for i in [0,1,2]], [np.array(data_y[i]) for i in [0,1,2]]
test_x, test_y = np.array(data_x[3]), np.array(data_y[3])
real_capacity = data[3][:, -1]


# Define model and fit data
if model_name == 'GPR':
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    kernel = DotProduct() + WhiteKernel()
    model = GaussianProcessRegressor(kernel=kernel, random_state=random_seed)
else:
    model = build_model(model_name)
criterion = build_criterion(criterion)

for i in range(len(train_x)):
    model.fit(train_x[i], train_y[i])

# test
pred_y = model.predict(test_x)
print('pred_y: ', pred_y.shape)

# Calculate the error
error = []
for i in range(len(test_y)):
    print(test_y[i], pred_y[i])
    error.append(criterion(test_y[i], pred_y[i]))
print('error: ', error)
print('mean error: ', np.mean(error))


def iterative_pred(model, input, EOL):
    # Iterative prediction
    input_list, pred_y = [input], []
    while True:
        pred_y.append(model.predict([input])[0])
        if all(pred_y[-1] > EOL):
            input = np.concatenate([input[len(pred_y[-1]):], pred_y[-1]])
            input_list.append(input)
        else:
            break
    # pred_y = np.concatenate(pred_y)
    return input_list, pred_y

class visual_iterative_pred():
    def __init__(self, model, input, start, EOL, real_capacity, ax):
        self.input_list, self.pred_y = iterative_pred(model, input, EOL)
        self.in_len, self.out_len, self.s = len(input), len(self.pred_y[0]), start
        capacity, = ax.plot(real_capacity, color='k', linewidth=2, markersize=3, marker='o', linestyle='dashed', label='Capacity')
        raw_input, = ax.plot(np.arange(self.s, self.s+len(input)), input, color='b', linewidth=2, marker='*', markersize=6, label='raw_input')
        EOL_im = ax.axhline(y=EOL, color='r', linestyle='dashed', linewidth=2, label='EOL')
        # self.input_im, = ax.plot(self.s*self.in_len, self.s*self.in_len+np.arange(len(input)), input, color='r', linewidth=2, marker='o', markersize=6, label='iter_input')
        self.input_im, = ax.plot([], [], color='r', linewidth=2, marker='o', markersize=6, label='iter_input')
        self.predicted, = ax.plot([], [], color='green', linewidth=3, markersize=12, label='predicted')
        self.pred_im, = ax.plot(np.arange(self.s+self.in_len, self.out_len + self.s+self.in_len), self.pred_y[0],
                                color='lime', linewidth=2, marker='*', markersize=4, label='predicting')
        ax.legend()
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Capacity(Ahr)')
        self.ims = [capacity, raw_input, EOL_im, self.pred_im, self.input_im]

    def my_update(self, i):
        self.input_im.set_data(np.arange(self.s+i*8, self.s+i*8+self.in_len), self.input_list[i])
        self.pred_im.set_data(np.arange(self.s+i * self.out_len + self.in_len, self.s+(i + 1) * self.out_len + self.in_len),
                              self.pred_y[i])
        if i > 0:
            self.predicted.set_data(np.arange(self.s+self.in_len, self.s+i*self.out_len+self.in_len), np.concatenate(self.pred_y[:i]))

        return self.ims


fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
start_cycle = 0
scope = visual_iterative_pred(model, test_x[start_cycle], start_cycle, 1.4, real_capacity, ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig, scope.my_update, range(len(scope.pred_y)), interval=250, blit=True)
ani.save(f'dynamic_{model_name}_{start_cycle}.gif', writer='pillow')
# ani.save('dynamic.gif', writer='imagemagick', fps=60)


if __name__ == '__main__':
    pass
