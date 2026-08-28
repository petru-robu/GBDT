#include "../inc/Dataset.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <unordered_set>

double Dataset::get_value(size_t row, size_t col) {
    return data[row * num_cols + col];
}

double Dataset::get_target(size_t row) {
    return target[row];
}

void Dataset::append_row_features(std::vector<double> row) {
    for (auto& val : row)
        data.push_back(val);
}

void Dataset::append_row_target(double val) {
    target.push_back(val);
}

int Dataset::get_feat_idx(std::string feat_name) {
    for (size_t idx = 0; idx < feat_names.size(); idx++) {
        if (feat_names[idx] == feat_name) return idx;
    }
    return -1;
}

void Dataset::display(int head) {
    if (head == -1 || head > (int)num_rows) head = num_rows;
    if (head == 0) {
        std::cout << "Empty Dataset\n";
        return;
    }

    std::vector<int> col_widths(num_cols, 0);
    for (size_t j = 0; j < num_cols; j++) {
        int max_w = feat_names[j].length();
        for (size_t i = 0; i < (size_t)head; i++) {
            std::ostringstream oss; oss << get_value(i, j);
            max_w = std::max(max_w, static_cast<int>(oss.str().length()));
        }
        col_widths[j] = max_w + 2;
    }

    int target_width = target_name.length();
    for (size_t i = 0; i < (size_t)head; i++) {
        std::ostringstream oss; oss << get_target(i);
        target_width = std::max(target_width, static_cast<int>(oss.str().length()));
    }
    target_width += 2;

    std::ostringstream idx_oss; idx_oss << (head - 1) << ": ";
    int idx_width = idx_oss.str().length();

    std::cout << std::string(idx_width, ' ');
    for (size_t j = 0; j < num_cols; j++) {
        std::cout << std::right << std::setw(col_widths[j]) << feat_names[j];
    }
    std::cout << std::right << std::setw(target_width) << target_name << '\n';

    for (size_t i = 0; i < (size_t)head; i++) {
        std::ostringstream row_prefix; row_prefix << i << ": ";
        std::cout << std::left << std::setw(idx_width) << row_prefix.str();
        for (size_t j = 0; j < num_cols; j++) {
            std::cout << std::right << std::setw(col_widths[j]) << get_value(i, j);
        }
        std::cout << std::right << std::setw(target_width) << get_target(i) << "\n";
    }
    std::cout << "\n";
}

void Dataset::drop_column(const std::string& col_name) {
    int drop_idx = -1;
    for (size_t i = 0; i < feat_names.size(); ++i) {
        if (feat_names[i] == col_name) {
            drop_idx = i; break;
        }
    }

    if (drop_idx == -1) {
        if (col_name == target_name) std::cerr << "Warning: Cannot drop target.\n";
        else std::cerr << "Warning: Column '" << col_name << "' not found.\n";
        return;
    }

    feat_names.erase(feat_names.begin() + drop_idx);
    std::vector<double> new_data;
    new_data.reserve(num_rows * (num_cols - 1));

    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t c = 0; c < num_cols; ++c) {
            if (c != static_cast<size_t>(drop_idx)) {
                new_data.push_back(get_value(r, c));
            }
        }
    }
    data = std::move(new_data);
    num_cols -= 1;
}

Dataset Dataset::get_datarow(size_t row_idx) {
    Dataset new_ds;

    if (row_idx >= num_rows) {
        std::cout << "Invalid row_idx!\n";
        return new_ds;
    }

    // 1. Copy the column names
    new_ds.feat_names = this->feat_names;
    new_ds.target_name = this->target_name;

    // 2. Copy the data
    for (size_t j = 0; j < num_cols; j++) {
        new_ds.data.push_back(get_value(row_idx, j));
    }

    // 3. Copy the target
    new_ds.append_row_target(get_target(row_idx));

    // 4. Update dimensions
    new_ds.num_rows = 1;
    new_ds.num_cols = this->num_cols;

    return new_ds;
}

Dataset Dataset::get_subset(const std::vector<size_t>& indices) {
    Dataset subset;
    
    // 1. Copy metadata to prevent segfaults!
    subset.feat_names = this->feat_names;
    subset.target_name = this->target_name;
    subset.num_cols = this->num_cols;
    subset.num_rows = indices.size();

    // 2. Copy the requested rows
    for (size_t row_idx : indices) {
        for (size_t j = 0; j < num_cols; j++) {
            subset.data.push_back(this->get_value(row_idx, j));
        }
        subset.append_row_target(this->get_target(row_idx));
    }
    
    return subset;
}

std::vector<Dataset> Dataset::get_rows() {
    std::vector<Dataset> rows;
    for(size_t i = 0; i < num_rows; i++) {
        rows.push_back(get_datarow(i));
    }
    return rows;
}

// OTHER DATASET RELATED FUNCTIONS
std::vector<std::string> single_sep_split_strip(std::string line, char sep) {
    size_t len = line.size();
    std::string curr = "";
    std::vector<std::string> tokens;

    for (size_t i = 0; i < len; i++) {
        if (line[i] == sep) {
            tokens.push_back(curr);
            curr = "";
        }
        else curr += line[i];
    }
    if (!curr.empty()) tokens.push_back(curr);

    for (auto& tok : tokens) {
        std::string stripped_tok = "";
        for (auto& ch : tok) {
            if (ch != ' ') stripped_tok.push_back(ch);
        }
        tok = stripped_tok;
    }
    return tokens;
}

Dataset load_csv(const std::string& csv_path, bool has_header, int target_col_idx, const std::vector<std::string>& drop_cols) {
    std::ifstream file(csv_path);
    if (!file.is_open()) throw std::runtime_error("Could not open file " + csv_path);

    Dataset dataset;
    std::string line;
    int row_count = 0;
    
    // Store indices of columns we want to ignore
    std::unordered_set<int> drop_indices;

    while (std::getline(file, line)) {
        row_count++;
        if (line.empty()) continue;

        std::vector<std::string> tokens = single_sep_split_strip(line, ',');
        
        // Default target column to the last column if not specified
        if (target_col_idx == -1) target_col_idx = tokens.size() - 1;

        // 1. Process Header
        if (has_header) {
            has_header = false;
            
            for (size_t i = 0; i < tokens.size(); i++) {
                std::string col_name = tokens[i];
                
                // Clean any carriage returns off the header name
                col_name.erase(std::remove(col_name.begin(), col_name.end(), '\r'), col_name.end());

                // If column name is in drop_cols, save its index to skip later
                if (std::find(drop_cols.begin(), drop_cols.end(), col_name) != drop_cols.end()) {
                    drop_indices.insert(i);
                    continue;
                }
                
                if ((int)i == target_col_idx) {
                    dataset.target_name = col_name + "(y)";
                } else {
                    dataset.feat_names.push_back(col_name);
                }
            }
            continue; // Move to the first data row
        }

        // 2. Process Data Rows
        std::vector<double> row_without_target;
        double target = 0.0;

        for (size_t i = 0; i < tokens.size(); i++) {
            // Skip this column entirely if it's in our drop list
            if (drop_indices.find(i) != drop_indices.end()) {
                continue;
            }

            std::string token = tokens[i];
            
            // Clean formatting artifacts
            token.erase(std::remove(token.begin(), token.end(), '\"'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());

            double val = 0.0; 

            if (!token.empty() && token != "NA") {
                try {
                    val = std::stod(token);
                } catch (const std::exception& e) {
                    throw std::runtime_error(
                        "Failed to parse numeric value at Row " + std::to_string(row_count) + 
                        ", Column " + std::to_string(i) + ". Value was: '" + token + "'"
                    );
                }
            }

            // Assign to target or features based on index
            if ((int)i == target_col_idx) {
                target = val;
            } else {
                row_without_target.push_back(val);
            }
        }

        dataset.num_cols = row_without_target.size();
        dataset.append_row_features(row_without_target);
        dataset.append_row_target(target);
        dataset.num_rows += 1;
    }
    return dataset;
}

double calculate_susbset_rss(Dataset& ds, const std::vector<size_t>& indices, double mean_value) {
    double rss = 0;
    for(auto &idx : indices) {
        double y = ds.get_target(idx);
        rss += (y - mean_value) * (y - mean_value);
    }
    return rss;
}

std::pair<Dataset, Dataset> train_test_split_ds(Dataset &ds, double train_pct) {
    if (train_pct <= 0.0 || train_pct >= 1.0) {
        std::cout << "Invalid percentage!\n";
        return {};
    }

    // 1. Create a vector of all row indices [0, 1, 2, ..., num_rows-1]
    std::vector<size_t> indices(ds.num_rows);
    std::iota(indices.begin(), indices.end(), 0);

    // 2. Shuffle the indices randomly
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // 3. Calculate how many rows go to the training set
    size_t train_size = static_cast<size_t>(ds.num_rows * train_pct);

    // 4. Split the indices into train and test vectors
    std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> test_indices(indices.begin() + train_size, indices.end());

    // 5. Generate the actual Dataset objects using get_subset
    Dataset train_ds = ds.get_subset(train_indices);
    Dataset test_ds = ds.get_subset(test_indices);

    return {train_ds, test_ds};
}

void Dataset::swap_in_target(std::vector<double> target_row) {
    if (target_row.size() != this->num_rows) {
        std::cout << "Error: New target size (" << target_row.size() << ") does not match dataset row count (" << this->num_rows << ").\n";
        return;
    }
    
    this->target = std::move(target_row);
}