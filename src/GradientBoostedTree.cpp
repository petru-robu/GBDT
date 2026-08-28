#include "../inc/GradientBoostedTree.hpp"
#include <iostream>

GradientBoostedTree::GradientBoostedTree(Dataset& ds, int max_depth, int minimum_region_samples, size_t n_estimators, double learning_rate):
n_estimators(n_estimators), learning_rate(learning_rate) {
    base_prediction = get_avg_target(ds);
    
    std::vector<double> true_y;
    for(size_t i = 0; i < ds.num_rows; i++) {
        true_y.push_back(ds.get_target(i));
    }

    std::vector<double> current_predictions(ds.num_rows, base_prediction);

    for(size_t tree_iter = 0; tree_iter < n_estimators; tree_iter++) {
        // std::cout << "Training Tree no. " << tree_iter << "...\n";

        std::vector<double> residuals;
        for(size_t i = 0; i < ds.num_rows; i++) {
            residuals.push_back(true_y[i] - current_predictions[i]);
        }

        // train the tree on the residuals
        ds.swap_in_target(residuals);
        DecisionTree tree(ds, max_depth, minimum_region_samples);

        for(size_t i = 0; i < ds.num_rows; i++) {
            double tree_pred = tree.predict(ds, i);
            current_predictions[i] += learning_rate * tree_pred;
        }

        // save the tree in the ensemble! 
        // use std::move for unique_ptrs inside decision tree
        trees.push_back(std::move(tree)); 
    }

    ds.swap_in_target(true_y);
}

double GradientBoostedTree::predict(Dataset& ds, size_t row_idx) {
    double final_prediction = base_prediction;

    for (size_t i = 0; i < trees.size(); i++) {
        double tree_pred = trees[i].predict(ds, row_idx);
        final_prediction += learning_rate * tree_pred;
    }

    return final_prediction;
}

std::vector<double> GradientBoostedTree::predict_all(Dataset& ds) {
    std::vector<double> all_predictions;
    all_predictions.reserve(ds.num_rows);

    for (size_t i = 0; i < ds.num_rows; i++) {
        all_predictions.push_back(this->predict(ds, i));
    }

    return all_predictions;
}