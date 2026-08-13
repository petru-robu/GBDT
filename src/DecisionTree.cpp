#include "../inc/DecisionTree.hpp"
#include <iostream>
#include <algorithm>

DecisionTree::DecisionTree(Dataset& ds, int max_depth) : max_depth(max_depth) {
    std::vector<size_t> all_indices;
    for(size_t i = 0; i < ds.num_rows; i++) {
        all_indices.push_back(i);
    }
    root = build_tree(ds, all_indices);
}

std::pair<std::string, double> DecisionTree::find_best_split(Dataset &ds, std::vector<size_t> indices) {
    double max_score_overall = -1, best_split_overall = -1;
    std::string best_feat = "";

    for(auto &feat: ds.feat_names) {
        size_t feat_idx = ds.get_feat_idx(feat);
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            return ds.get_value(i, feat_idx) < ds.get_value(j, feat_idx);
        }); 
        
        double total_sum = 0, total_count = (double)indices.size();
        for(auto &row_idx : indices) total_sum += ds.get_target(row_idx);

        double left_sum = 0, left_count = 0;
        double max_score_feat = -1, best_split_val_feat = -1;
        
        for(size_t i = 0; i < indices.size() - 1; i++) {
            size_t row_idx = indices[i];
            size_t next_row_idx = indices[i + 1];

            double curr_feat_val = ds.get_value(row_idx, feat_idx);
            double next_feat_val = ds.get_value(next_row_idx, feat_idx);
            double target_val = ds.get_target(row_idx);
            
            left_sum += target_val;
            left_count += 1;

            if(curr_feat_val == next_feat_val) continue;

            double right_sum = total_sum - left_sum;
            double right_count = total_count - left_count;

            double left_avg = left_sum / left_count;
            double right_avg = right_sum / right_count;
            double score = left_avg * left_sum + right_avg * right_sum;

            if(max_score_feat < score) {
                max_score_feat = score;
                best_split_val_feat = curr_feat_val;
            }
        }

        if(max_score_overall < max_score_feat) {
            max_score_overall = max_score_feat;
            best_split_overall = best_split_val_feat;
            best_feat = feat;
        }
    }
    return {best_feat, best_split_overall};
}

double DecisionTree::calculate_leaf_value(Dataset& ds, std::vector<size_t> indices) {
    if (indices.empty()) return 0.0;
    double total_target = 0;
    for(size_t i = 0; i < indices.size(); i++) {
        total_target += ds.get_target(indices[i]);
    }
    return total_target / indices.size();
}

std::unique_ptr<Node> DecisionTree::build_tree(Dataset& ds, std::vector<size_t> indices, int depth) {
    std::unique_ptr<Node> node = std::make_unique<Node>();

    if (depth >= max_depth || indices.size() <= 2) { 
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

    std::cout << "This node is split on: " << best_feat << " > " << threshold << "\n";

    for(size_t i = 0; i < indices.size(); i++) {
        size_t row_idx = indices[i];
        double curr_feat_val = ds.get_value(row_idx, ds.get_feat_idx(best_feat));  
        if (curr_feat_val > threshold) {
            right_indices.push_back(row_idx); 
        } else {
            left_indices.push_back(row_idx); 
        }
    }

    node->feature_name = best_feat;
    node->threshold = threshold;
    node->left = build_tree(ds, left_indices, depth + 1);
    node->right = build_tree(ds, right_indices, depth + 1);

    return node;
}