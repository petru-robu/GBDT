#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <iomanip>
#include <sstream>
#include <algorithm>

struct Dataset {
    std::vector<double> data; // flattened features
    std::vector<std::string> feat_names;
    std::string target_name;
    std::vector<double> target; 

    size_t num_rows;
    size_t num_cols;

    double get_value(size_t row, size_t col) {
        return data[row * num_cols + col];
    }

    double get_target(size_t row) {
        return target[row];
    }

    void append_row_features(std::vector<double> row) {
        for(auto &val: row) {
            data.push_back(val);
        }
    }

    void append_row_target(double val) {
        target.push_back(val);
    }

    size_t get_feat_idx(std::string feat_name) {
        for (size_t idx = 0; idx < feat_names.size(); idx++) {
            if(feat_names[idx] == feat_name)
                return idx;
        }
        return -1;
    }

    void display(int head = -1) {
        if(head == -1 || head > num_rows) {
            head = num_rows;
        }
        
        if (head == 0) {
            std::cout << "Empty Dataset\n";
            return;
        }

        // 1. Calculate the required width for each feature column
        std::vector<int> col_widths(num_cols, 0);
        for(size_t j = 0; j < num_cols; j++) {
            int max_w = feat_names[j].length(); 
            for(size_t i = 0; i < (size_t)head; i++) {
                std::ostringstream oss;
                oss << get_value(i, j);
                max_w = std::max(max_w, static_cast<int>(oss.str().length()));
            }
            col_widths[j] = max_w + 2; 
        }

        // Calculate the required width for the target column
        int target_width = target_name.length();
        for(size_t i = 0; i < (size_t)head; i++) {
            std::ostringstream oss;
            oss << get_target(i);
            target_width = std::max(target_width, static_cast<int>(oss.str().length()));
        }
        target_width += 2; // Add padding

        // Calculate the width for the row index column
        std::ostringstream idx_oss;
        idx_oss << (head - 1) << ": ";
        int idx_width = idx_oss.str().length();

        // 2. Print Header
        std::cout << std::string(idx_width, ' '); 
        for(size_t j = 0; j < num_cols; j++) {
            std::cout << std::right << std::setw(col_widths[j]) << feat_names[j]; 
        }
        // Print Target Header
        std::cout << std::right << std::setw(target_width) << target_name << '\n';

        // 3. Print Data Rows
        for(size_t i = 0; i < (size_t)head; i++) {
            std::ostringstream row_prefix;
            row_prefix << i << ": ";
            std::cout << std::left << std::setw(idx_width) << row_prefix.str();
            
            // Print Feature Values
            for(size_t j = 0; j < num_cols; j++) {
                std::cout << std::right << std::setw(col_widths[j]) << get_value(i, j);
            }
            // Print Target Value
            std::cout << std::right << std::setw(target_width) << get_target(i) << "\n";
        }

        std::cout << "\n";
    }

    void drop_column(const std::string& col_name) {
        // 1. Find the index of the feature to drop
        int drop_idx = -1;
        for (size_t i = 0; i < feat_names.size(); ++i) {
            if (feat_names[i] == col_name) {
                drop_idx = i;
                break;
            }
        }

        // Safety check if column doesn't exist or if they tried to drop the target
        if (drop_idx == -1) {
            if (col_name == target_name) {
                std::cerr << "Warning: Cannot drop the target column using this method.\n";
            } else {
                std::cerr << "Warning: Column '" << col_name << "' not found.\n";
            }
            return;
        }

        // 2. Remove the name from feat_names
        feat_names.erase(feat_names.begin() + drop_idx);

        // 3. Rebuild the flattened data vector without the dropped column
        std::vector<double> new_data;
        new_data.reserve(num_rows * (num_cols - 1)); // Pre-allocate memory for performance

        for (size_t r = 0; r < num_rows; ++r) {
            for (size_t c = 0; c < num_cols; ++c) {
                if (c != static_cast<size_t>(drop_idx)) {
                    new_data.push_back(get_value(r, c));
                }
            }
        }

        // Replace the old data with the newly built vector
        data = std::move(new_data);
        
        // 4. Update the column count
        num_cols -= 1;
    }
};

std::vector<std::string> single_sep_split_strip(std::string line, char sep) {
    size_t len = line.size();
    std::string curr = "";
    std::vector<std::string> tokens;    

    // split by separator
    for(size_t i = 0; i < len; i++) {
        if(line[i] == sep) {
            tokens.push_back(curr);
            curr = "";
        }
        else {
            curr += line[i];
        }
    }
    if(!curr.empty())
        tokens.push_back(curr);

    // strip the tokens
    for(auto &tok:tokens) {
        std::string stripped_tok = "";
        for(auto &ch:tok) {
            if (ch != ' ') {
                stripped_tok.push_back(ch);
            }
        }
        tok = stripped_tok;
    }

    return tokens;
}


Dataset load_csv(const std::string& csv_path, bool has_header, int target_col_idx = -1) {
    std::ifstream file(csv_path);
    if(!file.is_open()) {
        throw std::runtime_error("Could not open file " + csv_path);
    }

    Dataset dataset;
    std::string line;

    dataset.num_rows = 0;
    dataset.num_cols = 0;

    while(std::getline(file, line)) {
        if(line.empty())
            continue;

        if(has_header) {
            has_header = false;
            std::vector<std::string> tokens = single_sep_split_strip(line, ',');
            std::vector<std::string> feature_names;

            size_t rs = tokens.size();
            if(target_col_idx == -1) 
                target_col_idx = rs - 1;
            
            for(size_t i = 0; i < tokens.size(); i++) {
                if(target_col_idx == i) {
                    dataset.target_name = tokens[i] + "(y)";
                }
                else {
                    feature_names.push_back(tokens[i]);
                }
            }

            dataset.feat_names = feature_names;
            continue;
        }

        // split by comma
        std::vector<std::string> tokens = single_sep_split_strip(line, ',');
        size_t rs = tokens.size();
        if(target_col_idx == -1) 
            target_col_idx = rs - 1;

        // cast to double
        std::vector<double> row;
        for(auto &tok:tokens) {
            row.push_back(stod(tok));
        }   

        std::vector<double> row_without_target;
        double target;

        for (size_t i = 0; i < row.size(); i++) {
            if (i == target_col_idx) {
                target = row[i];
            }
            else {
                row_without_target.push_back(row[i]);
            }
        }

        // update dataset
        dataset.num_cols = row_without_target.size();
        dataset.append_row_features(row_without_target);
        dataset.append_row_target(target);
        dataset.num_rows += 1;
    }
    return dataset;
}


/*
Decision Trees work by selecting the best attribute at each step to split the data. 
This selection is based on statistical metrics that measure data impurity or uncertainty.

Start with the full dataset as the root node.
Select the best feature using a splitting criterion.
Split the dataset into subsets.
Repeat the process recursively until stopping conditions are met.
Assign class labels at leaf nodes.
*/

// this is classic binary tree stuff
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

    std::pair<std::string, double> find_best_split(Dataset &ds, std::vector<size_t> indices) {
        double max_score_overall = -1, best_split_overall = -1;
        std::string best_feat = "";

        for(auto &feat: ds.feat_names) {
            // std::cout << "Best split - " << feat << "\n";
            size_t feat_idx = ds.get_feat_idx(feat);

            // sort indices by this feature
            std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {return ds.get_value(i, feat_idx) < ds.get_value(j, feat_idx);}); 
            
            double total_sum = 0, total_count = (double)indices.size();
            for(auto &row_idx : indices)
                total_sum += ds.get_target(row_idx);

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


                if(curr_feat_val == next_feat_val) {
                    continue;
                }

                // std::cout << "Testing split on " << feat << " > " << curr_feat_val << "\n";

                double right_sum = total_sum - left_sum;
                double right_count = total_count - left_count;

                double left_avg = left_sum / left_count;
                double right_avg = right_sum / right_count;
                double score = left_avg * left_sum + right_avg * right_sum;
                
                // std::cout << "Score: " << score << "\n\n";

                if(max_score_feat < score) {
                    max_score_feat = score;
                    best_split_val_feat = curr_feat_val;
                }
            }

            // std::cout << "For feat " << feat <<" best split val is: " << best_split_val_feat << "\n\n"; 

            if(max_score_overall < max_score_feat) {
                max_score_overall = max_score_feat;
                best_split_overall = best_split_val_feat;
                best_feat = feat;
            }
        }

        // std::cout << "Best split is: " << best_feat << " > " << best_split_overall;
        return {best_feat, best_split_overall};
    }

    double calculate_leaf_value(Dataset& ds, std::vector<size_t> indices) {
        if (indices.empty()) 
            return 0.0;

        size_t count = indices.size();
        double total_target = 0;
        for(size_t i = 0; i < count; i++) {
            size_t row_idx = indices[i];
            total_target += ds.get_target(row_idx);
        }
        return total_target / count;
    }

    std::unique_ptr<Node> build_tree(Dataset& ds, std::vector<size_t> indices, int depth = 0) {
        std::unique_ptr<Node> node = std::make_unique<Node>();

        if (depth >= max_depth || indices.size() <= 2) { 
            node->is_leaf = true;
            node->value = calculate_leaf_value(ds, indices);
            return node;
        }

        auto [best_feat, threshold] = find_best_split(ds, indices);
        std::vector<size_t> right_indices, left_indices; 

        // no valid split found
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
                right_indices.push_back(i); 
            }
            else {
                left_indices.push_back(i); 
            }
        }

        node->feature_name = best_feat;
        node->threshold = threshold;
        node->left = build_tree(ds, left_indices, depth + 1);
        node->right = build_tree(ds, right_indices, depth + 1);

        return node;
    }

public:
    DecisionTree(Dataset& ds, int max_depth): max_depth(max_depth) {
        std::vector<size_t> all_indices;
        for(int i = 0; i < ds.num_rows; i++) {
            all_indices.push_back(i);
        }
        root = build_tree(ds, all_indices);
    }
};


int main() {    
    Dataset ds = load_csv("data.csv", true);
    ds.drop_column("id");
    ds.display();
    
    DecisionTree dt(ds, 4);


    return 0;
}