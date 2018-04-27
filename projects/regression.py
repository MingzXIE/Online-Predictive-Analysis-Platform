import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor


# Store the algorithms into a dictionary
reg_all= {'Linear Regression':LinearRegression(),
          'Support Vector Machine':SVR(),
          'Byesian Ridge':BayesianRidge(),
          'Lasso':Lasso(),
          'K Neighbors Regression':KNeighborsRegressor(n_neighbors =2)
          }


# Read Data
def read_file(filename, n_fold_input = 5):
    data = pd.read_csv(filename, sep='\t', header=None)
    data_x = data.iloc[:, :-1]
    data_y = data.iloc[:, -1]
    n_fold = n_fold_input  # n_fold
    # Split data by KFold
    kf = KFold(len(data_y), n_fold)

    return data_x, data_y, kf


def regression_test(reg, kf, n_fold, data_x, data_y):

    # reg represents the type of the regression algorithm
    # adopt k-fold cross validation to test get the accuracy of each algorithms

    list_result = []
    for train_index,test_index in kf:
        train_x,train_y = data_x.iloc[train_index],data_y.iloc[train_index] # training data
        test_x,test_y = data_x.iloc[test_index],data_y.iloc[test_index] # testing data

        reg.fit(train_x,train_y)
        # make prediction
        predict_y = reg.predict(test_x)
        # test
        list_result.append(mean_squared_error(test_y,predict_y))
    # return the average error rate of each algorithm
    return sum(list_result)/n_fold


# select the best algorithm/classifier for the user
def select_best_algorithm(filename, n_fold_input = 5):
    reg_all_label = []  # The name of the algorithms
    reg_all_result = []  # The result of tresting
    squared_error_list = ''
    matrix, labels, kf = read_file(filename, n_fold_input)
    for k, v in reg_all.items():
        # Training and Testing
        reg_all_label.append(k)
        squared_error_tmp = regression_test(v, kf, n_fold_input, matrix, labels)
        reg_all_result.append(squared_error_tmp)
        squared_error_list += str(k) + ": " + str(squared_error_tmp) + "\n"
    # select best algorithm via the accuracy
    best_choice_index = reg_all_result.index(min(reg_all_result))
    best_algorithm = reg_all_label[best_choice_index]
    lowest_squared_error = min(reg_all_result)
    print(lowest_squared_error)
    print(squared_error_list)
    print(best_algorithm)
    return lowest_squared_error, squared_error_list, best_algorithm


def input_and_predict(best_algorithm, unlabeled_data):
    reg_best = reg_all[best_algorithm]
    data_to_predict = np.array(string_to_list(unlabeled_data)).reshape(1, -1)
    result = reg_best.predict(data_to_predict)
    print(reg_best)
    return result[0]


def string_to_list(string_input):
    line = string_input.strip()
    float_list=[]
    elements_list = line.split(',')
    for str_number in elements_list:
        float_num = float(str_number)
        float_list.append(float_num)
    return float_list
