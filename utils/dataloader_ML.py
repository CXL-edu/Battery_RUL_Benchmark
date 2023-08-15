import os
import numpy as np
import pandas as pd



def load_NASA_data(data_source, seq_len, pred_len):
    data_path = 'data/NASA/'
    data_list_npy = 'NASA'
    data_list_csv = ['B0005', 'B0006', 'B0007', 'B0018']

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