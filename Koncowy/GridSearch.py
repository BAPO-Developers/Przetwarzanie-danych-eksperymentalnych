import numpy as np
from sklearn.tree import DecisionTreeRegressor
from Goal import goal_function
from sklearn.linear_model import Ridge
import tqdm


class MyGridSearch:
    def __init__(self, dict, clf, X, y, seed=1410):
        parameters_number = len(dict)
        comb_numb_temp = np.zeros((parameters_number))
        all_params = []

        for i, param_key in enumerate(dict):
            comb_numb_temp[i] = len(dict[param_key])
            all_params.append(dict[param_key])
        self.combination_number = int(np.prod(comb_numb_temp))
        self.combination = np.zeros(((self.combination_number, parameters_number))).astype(int)


        for i, zero_level_params in enumerate(all_params[0]):
            combination_in_lower_level  = int(self.combination_number/len(all_params[0]))
            combination_zero_level = combination_in_lower_level
            if 1 == parameters_number:
                self.combination[i, 0] = int(i)
            else:
                self.combination[combination_in_lower_level*i:combination_in_lower_level*(i+1), 0] = int(i)
                for j in range(1, parameters_number):
                    combination_in_lower_level = int(combination_in_lower_level/len(all_params[j]))

                    if j == 1:
                        repeat = 1
                    else:
                        repeat = len(all_params[1])
                        for m in range(3, j + 1):
                            repeat *= len(all_params[m-1])
                    for l in range(repeat):
                        l_rep = l
                        if repeat == 1:
                            l_rep = 0
                        for k in range(len(all_params[j])):
                            start = combination_zero_level*i+combination_in_lower_level*(k+l_rep*len(all_params[j]))
                            stop = combination_zero_level*i+combination_in_lower_level*(k+1+l_rep*len(all_params[j]))
                            self.combination[start:stop, j] = int(k)
        print(self.combination.dtype)
        self.all_grid_dict = []
        for point_on_grid in tqdm.tqdm(self.combination):
            grid_dict = {}
            for i, param_name in enumerate(dict):
                grid_dict[param_name]=all_params[i][point_on_grid[i]]
#             self.all_grid_dict.append((grid_dict, goal_function(clf.__init__(**grid_dict), X, y, seed)))
            clf.__init__(**grid_dict)
            self.all_grid_dict.append((grid_dict, goal_function(clf, X, y, seed)))
        self.best_index = 0
        self.best_mse = self.all_grid_dict[0][1]
        for i, grid_dict in enumerate(self.all_grid_dict):
            if grid_dict[1] < self.best_mse:
                self.best_index = i
                self.best_mse = grid_dict[1]
    
    def get_best(self):
        return self.all_grid_dict[self.best_index]
        
