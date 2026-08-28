#ifndef GRADIENTBOOSTEDTREE_HPP
#define GRADIENTBOOSTEDTREE_HPP

#include "DecisionTree.hpp"
#include "DecisionTreeUtils.hpp"
#include <string>
#include <vector>

class GradientBoostedTree {
private:
    std::vector<DecisionTree> trees;
    double base_prediction;
    double learning_rate;
    size_t n_estimators; // this is the no. of trees

public:
    GradientBoostedTree(Dataset& ds, int max_depth, int minimum_region_samples, size_t n_estimators, double learning_rate);
    double predict(Dataset& ds, size_t row_idx);
    std::vector<double> predict_all(Dataset& ds);
};


#endif