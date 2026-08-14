#include "../inc/DecisionTree.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>

DecisionTree::DecisionTree(Dataset& ds, int max_depth, int minimum_region_samples) : max_depth(max_depth), minimum_region_samples(minimum_region_samples) {
    std::vector<size_t> all_indices;
    for (size_t i = 0; i < ds.num_rows; i++) {
        all_indices.push_back(i);
    }
    root = build_tree(ds, all_indices);
}

std::pair<std::string, double> DecisionTree::find_best_split(Dataset& ds, std::vector<size_t> indices) {
    double max_score_overall = -1, best_split_overall = -1;
    std::string best_feat = "";

    for (auto& feat : ds.feat_names) {
        size_t feat_idx = ds.get_feat_idx(feat);
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            return ds.get_value(i, feat_idx) < ds.get_value(j, feat_idx);
        });

        double total_sum = 0, total_count = (double)indices.size();
        for (auto& row_idx : indices) 
            total_sum += ds.get_target(row_idx);

        double left_sum = 0, left_count = 0;
        double max_score_feat = -1, best_split_val_feat = -1;

        for (size_t i = 0; i < indices.size() - 1; i++) {
            size_t row_idx = indices[i];
            size_t next_row_idx = indices[i + 1];

            double curr_feat_val = ds.get_value(row_idx, feat_idx);
            double next_feat_val = ds.get_value(next_row_idx, feat_idx);
            double target_val = ds.get_target(row_idx);

            left_sum += target_val;
            left_count += 1;

            if (curr_feat_val == next_feat_val) continue;

            double right_sum = total_sum - left_sum;
            double right_count = total_count - left_count;

            double left_avg = left_sum / left_count;
            double right_avg = right_sum / right_count;
            double score = left_avg * left_sum + right_avg * right_sum;

            if (max_score_feat < score) {
                max_score_feat = score;
                best_split_val_feat = curr_feat_val;
            }
        }

        if (max_score_overall < max_score_feat) {
            max_score_overall = max_score_feat;
            best_split_overall = best_split_val_feat;
            best_feat = feat;
        }
    }
    return { best_feat, best_split_overall };
}

double DecisionTree::calculate_leaf_value(Dataset& ds, std::vector<size_t> indices) {
    if (indices.empty()) return 0.0;
    double total_target = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        total_target += ds.get_target(indices[i]);
    }
    return total_target / indices.size();
}

std::unique_ptr<Node> DecisionTree::build_tree(Dataset& ds, std::vector<size_t> indices, int depth) {
    std::unique_ptr<Node> node = std::make_unique<Node>();

    // For pruning: 
    // 1. Calculate the node as if were a leaf right now:
    node->value = calculate_leaf_value(ds, indices);
    node->node_error = calculate_susbset_rss(ds, indices, node->value);

    // stop if tree is too deep or if region contains less than minimum indices count
    if (depth >= max_depth || indices.size() <= minimum_region_samples) {
        node->is_leaf = true;
        node->value = calculate_leaf_value(ds, indices);
        return node;
    }

    auto [best_feat, threshold] = find_best_split(ds, indices);
    std::vector<size_t> right_indices, left_indices;

    if (best_feat == "") {
        node->is_leaf = true;
        node->value = calculate_leaf_value(ds, indices);
        return node;
    }

    // std::cout << "This node is split on: " << best_feat << " > " << threshold << "\n";

    for (size_t i = 0; i < indices.size(); i++) {
        size_t row_idx = indices[i];
        double curr_feat_val = ds.get_value(row_idx, ds.get_feat_idx(best_feat));
        if (curr_feat_val > threshold) {
            right_indices.push_back(row_idx);
        }
        else {
            left_indices.push_back(row_idx);
        }
    }

    node->feature_name = best_feat;
    node->threshold = threshold;
    node->left = build_tree(ds, left_indices, depth + 1);
    node->right = build_tree(ds, right_indices, depth + 1);

    // bottom-up sum the errors
    node->subtree_error = node->left->subtree_error + node->right->subtree_error;
    node->num_leaves += node->left->num_leaves + node->right->num_leaves;

    return node;
}

double DecisionTree::predict(Dataset& ds, size_t row_idx) {
    // check if tree is built
    if (root == nullptr) {
        std::cout << "Tree is not built!\n";
        return std::nan("");
    }

    Dataset row_to_predict = ds.get_datarow(row_idx);

    // check if features are the same
    if (row_to_predict.feat_names.size() != ds.feat_names.size()) {
        std::cout << "Different feature count in testing row!\n";
        return std::nan("");
    }

    for (auto& feat : row_to_predict.feat_names) {
        if (ds.get_feat_idx(feat) == -1) {
            std::cout << "Different features in testing row!\n";
            return std::nan("");
        }
    }

    // go on the tree
    Node* curr = root.get();

    while (curr && !curr->is_leaf) {
        std::string node_feat_name = curr->feature_name;
        double node_threshold = curr->threshold;

        double row_val = row_to_predict.get_value(0, row_to_predict.get_feat_idx(node_feat_name));

        if (row_val > node_threshold) {
            curr = curr->right.get();
        }
        else {
            curr = curr->left.get();
        }
    }

    if (!curr || !curr->is_leaf) {
        std::cout << "Error traversing tree!\n";
        return std::nan("");
    }

    return curr->value;
}

void DecisionTree::calculate_metrics(Node* node) {
    // Similar to what happens in build tree, but done again

    if(node == nullptr) {
        return;
    }

    if(node->is_leaf) {
        node->subtree_error = node->node_error;
        node->num_leaves = 1;
        return;
    }

    calculate_metrics(node->left.get());
    calculate_metrics(node->right.get());

    node->subtree_error = node->left->subtree_error + node->right->subtree_error;
    node->num_leaves = node->left->num_leaves + node->right->num_leaves;
}

Node* DecisionTree::find_weakest_link(Node* current_node, double &min_score, Node*& weakest_node) {
    if(current_node->is_leaf) {
        return nullptr;
    }

    // how much less error we get if we make the split vs just if we just close there and make the leaf
    double score = (current_node->node_error - current_node->subtree_error) / (current_node->num_leaves - 1);

    if (score < min_score) {
        min_score = score;
        weakest_node = current_node;
    }

    find_weakest_link(current_node->left.get(), min_score, weakest_node);
    find_weakest_link(current_node->right.get(), min_score, weakest_node);

    return weakest_node;
}

bool DecisionTree::prune_one_branch() {
    double min_score = 99999999;
    Node* weakest = nullptr;

    find_weakest_link(root.get(), min_score, weakest);

    if(weakest != nullptr) {
        weakest->is_leaf = true;
        weakest->right.reset();
        weakest->left.reset();

        // recalculate all metrics - tree shrank, other errors now.
        calculate_metrics(root.get());
        return true;
    }

    return false;
}