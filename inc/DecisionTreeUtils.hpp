#ifndef DECISIONTREEUTILS_HPP
#define DECISIONTREEUTILS_HPP

#include "Dataset.hpp"
#include "DecisionTree.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

double get_avg_target(Dataset &ds);

double calculate_mse(DecisionTree& dt, Dataset& test_ds);

void evaluate_split_dt(Dataset& ds, DecisionTree& dt);

double cross_validate(Dataset& ds, DecisionTree& tree, int k_folds);

int find_optimal_prunes(Dataset& ds);


#endif