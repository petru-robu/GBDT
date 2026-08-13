#include <iostream>
#include <vector>
#include <fstream>

struct Dataset {
    std::vector<double> data; // flattened features
    std::vector<double> target; 

    size_t num_rows;
    size_t num_cols;

    double get_value(size_t row, size_t col) {
        return data[col * num_rows + row];
    }

    double get_target(size_t row) {
        return target[row];
    }

    void append_row(std::vector<double> row) {
        for(auto &val: row) {
            data.push_back(val);
        }
    }

    void display(int head = -1) {
        if(head == -1) {
            head = num_rows;
        }
        std::cout << "---------DS---------\n";
        for(size_t i = 0; i < (size_t)head; i++) {
            std::cout<< i << ": ";
            for(size_t j = 0; j < num_cols; j++) {
                std::cout << get_value(i, j) << "   ";
            }
            std::cout << "\n";
        }

        std::cout<<"\n";
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


Dataset load_csv(const std::string& csv_path, bool has_header) {
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
            continue;
        }

        // split by comma
        std::vector<std::string> tokens = single_sep_split_strip(line, ',');

        // cast to double
        std::vector<double> row;
        for(auto &tok:tokens) {
            row.push_back(stod(tok));
        }   

        // update dataset
        dataset.num_cols = row.size();
        dataset.append_row(row);    
        dataset.num_rows += 1;
    }

    return dataset;
}


int main() {    
    Dataset ds = load_csv("data.csv", true);
    ds.display();

    return 0;
}