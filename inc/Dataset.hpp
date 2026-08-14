#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <iostream>

struct Dataset {
    std::vector<double> data; 
    std::vector<std::string> feat_names;
    
    std::string target_name;
    std::vector<double> target; 

    size_t num_rows = 0;
    size_t num_cols = 0;

    double get_value(size_t row, size_t col);
    double get_target(size_t row);
    void append_row_features(std::vector<double> row);
    void append_row_target(double val);
    int get_feat_idx(std::string feat_name);
    void display(int head = -1);
    void drop_column(const std::string& col_name);
    Dataset get_datarow(size_t row_idx);
    Dataset get_subset(const std::vector<size_t>& indices);
    std::vector<Dataset> get_rows();

};

// Helper functions
std::vector<std::string> single_sep_split_strip(std::string line, char sep);
Dataset load_csv(const std::string& csv_path, bool has_header, int target_col_idx = -1);
double calculate_susbset_rss(Dataset& ds, const std::vector<size_t>& indices, double mean_value);
std::pair<Dataset, Dataset> train_test_split_ds(Dataset &ds, double train_pct);



#endif