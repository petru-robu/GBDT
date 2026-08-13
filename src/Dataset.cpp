#include "../inc/Dataset.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>

double Dataset::get_value(size_t row, size_t col) {
    return data[row * num_cols + col];
}

double Dataset::get_target(size_t row) {
    return target[row];
}

void Dataset::append_row_features(std::vector<double> row) {
    for(auto &val: row) data.push_back(val);
}

void Dataset::append_row_target(double val) {
    target.push_back(val);
}

size_t Dataset::get_feat_idx(std::string feat_name) {
    for (size_t idx = 0; idx < feat_names.size(); idx++) {
        if(feat_names[idx] == feat_name) return idx;
    }
    return -1;
}

void Dataset::display(int head) {
    if(head == -1 || head > (int)num_rows) head = num_rows;
    if (head == 0) {
        std::cout << "Empty Dataset\n";
        return;
    }

    std::vector<int> col_widths(num_cols, 0);
    for(size_t j = 0; j < num_cols; j++) {
        int max_w = feat_names[j].length(); 
        for(size_t i = 0; i < (size_t)head; i++) {
            std::ostringstream oss; oss << get_value(i, j);
            max_w = std::max(max_w, static_cast<int>(oss.str().length()));
        }
        col_widths[j] = max_w + 2; 
    }

    int target_width = target_name.length();
    for(size_t i = 0; i < (size_t)head; i++) {
        std::ostringstream oss; oss << get_target(i);
        target_width = std::max(target_width, static_cast<int>(oss.str().length()));
    }
    target_width += 2; 

    std::ostringstream idx_oss; idx_oss << (head - 1) << ": ";
    int idx_width = idx_oss.str().length();

    std::cout << std::string(idx_width, ' '); 
    for(size_t j = 0; j < num_cols; j++) {
        std::cout << std::right << std::setw(col_widths[j]) << feat_names[j]; 
    }
    std::cout << std::right << std::setw(target_width) << target_name << '\n';

    for(size_t i = 0; i < (size_t)head; i++) {
        std::ostringstream row_prefix; row_prefix << i << ": ";
        std::cout << std::left << std::setw(idx_width) << row_prefix.str();
        for(size_t j = 0; j < num_cols; j++) {
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

std::vector<std::string> single_sep_split_strip(std::string line, char sep) {
    size_t len = line.size();
    std::string curr = "";
    std::vector<std::string> tokens;    

    for(size_t i = 0; i < len; i++) {
        if(line[i] == sep) {
            tokens.push_back(curr);
            curr = "";
        } else curr += line[i];
    }
    if(!curr.empty()) tokens.push_back(curr);

    for(auto &tok:tokens) {
        std::string stripped_tok = "";
        for(auto &ch:tok) {
            if (ch != ' ') stripped_tok.push_back(ch);
        }
        tok = stripped_tok;
    }
    return tokens;
}

Dataset load_csv(const std::string& csv_path, bool has_header, int target_col_idx) {
    std::ifstream file(csv_path);
    if(!file.is_open()) throw std::runtime_error("Could not open file " + csv_path);

    Dataset dataset;
    std::string line;

    while(std::getline(file, line)) {
        if(line.empty()) continue;

        if(has_header) {
            has_header = false;
            std::vector<std::string> tokens = single_sep_split_strip(line, ',');
            if(target_col_idx == -1) target_col_idx = tokens.size() - 1;
            
            for(size_t i = 0; i < tokens.size(); i++) {
                if((int)i == target_col_idx) dataset.target_name = tokens[i] + "(y)";
                else dataset.feat_names.push_back(tokens[i]);
            }
            continue;
        }

        std::vector<std::string> tokens = single_sep_split_strip(line, ',');
        if(target_col_idx == -1) target_col_idx = tokens.size() - 1;

        std::vector<double> row_without_target;
        double target = 0;

        for (size_t i = 0; i < tokens.size(); i++) {
            if ((int)i == target_col_idx) target = stod(tokens[i]);
            else row_without_target.push_back(stod(tokens[i]));
        }

        dataset.num_cols = row_without_target.size();
        dataset.append_row_features(row_without_target);
        dataset.append_row_target(target);
        dataset.num_rows += 1;
    }
    return dataset;
}