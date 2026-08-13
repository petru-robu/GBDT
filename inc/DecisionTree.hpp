/*
    This is actually a regression decision tree.
*/

#ifndef DECISIONTREE_HPP
#define DECISIONTREE_HPP

#include "Dataset.hpp"
#include <memory>
#include <string>
#include <vector>

struct Node {
    bool is_leaf = false;
    double value = 0.0;
    std::unique_ptr<Node> left, right;
    std::string feature_name;
    double threshold;
};

class DecisionTree {
private:
    std::unique_ptr<Node> root;
    int max_depth;

    std::pair<std::string, double> find_best_split(Dataset &ds, std::vector<size_t> indices);
    double calculate_leaf_value(Dataset& ds, std::vector<size_t> indices);
    std::unique_ptr<Node> build_tree(Dataset& ds, std::vector<size_t> indices, int depth = 0);

public:
    DecisionTree(Dataset& ds, int max_depth);
};

#endif