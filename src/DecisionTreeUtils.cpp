#include "../inc/DecisionTreeUtils.hpp"

double calculate_mse(DecisionTree& dt, Dataset& test_ds) {
    double total_squared_error = 0.0;
    for (size_t i = 0; i < test_ds.num_rows; ++i) {
        double pred = dt.predict(test_ds, i);
        double real = test_ds.get_target(i);
        total_squared_error += (pred - real) * (pred - real);
    }
    return (test_ds.num_rows > 0) ? (total_squared_error / test_ds.num_rows) : 0.0;
}

void evaluate_split_dt(Dataset& ds, DecisionTree& dt) {
    auto [train_ds, test_ds] = train_test_split_ds(ds, 0.6);

    size_t n = test_ds.num_rows;

    std::cout << '\n' << std::string(55, '-') << '\n';
    std::cout << std::left << std::setw(10) << "Row Index"
        << std::right << std::setw(15) << "Predicted (y)"
        << std::setw(15) << "Actual (y)"
        << std::setw(15) << "Difference" << '\n';
    std::cout << std::string(55, '-') << '\n';

    // Limit printing to the first 15 rows to prevent terminal lag
    size_t print_limit = std::min(n, static_cast<size_t>(15));
    for (size_t i = 0; i < print_limit; i++) {
        double pred = dt.predict(test_ds, i);
        double real = test_ds.get_target(i);
        double diff = pred - real;

        std::cout << std::left << std::setw(10) << i
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(15) << pred
            << std::setw(15) << real
            << std::showpos << std::setw(15) << diff << std::noshowpos << '\n';
    }

    if (n > print_limit) {
        std::cout << "... and " << (n - print_limit) << " more rows hidden.\n";
    }

    double mse = calculate_mse(dt, test_ds);

    std::cout << std::string(55, '-') << '\n';
    std::cout << "Evaluation Summary\n";
    std::cout << "Test Set Size : " << n << " samples\n";
    std::cout << "Final MSE     : " << std::fixed << std::setprecision(4) << mse << '\n';
    std::cout << std::string(55, '-') << "\n\n";
}

double cross_validate(Dataset& ds, DecisionTree& tree, int k_folds) {
    std::vector<size_t> indices(ds.num_rows);
    std::iota(indices.begin(), indices.end(), 0); // all indices

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    double total_mse = 0.0;
    size_t fold_size = ds.num_rows / k_folds;

    for (int k = 0; k < k_folds; k++) {
        size_t start_idx = k * fold_size;
        size_t end_idx = (k == k_folds - 1) ? ds.num_rows : start_idx + fold_size;

        std::vector<size_t> test_indices(indices.begin() + start_idx, indices.begin() + end_idx);
        std::vector<size_t> train_indices;
        train_indices.insert(train_indices.end(), indices.begin(), indices.begin() + start_idx);
        train_indices.insert(train_indices.end(), indices.begin() + end_idx, indices.end());

        Dataset train_set = ds.get_subset(train_indices);
        Dataset test_set = ds.get_subset(test_indices);

        double mse = calculate_mse(tree, test_set);

        std::cout << "Fold " << k + 1 << " MSE: " << mse << "\n";
        total_mse += mse;
    }

    return total_mse / k_folds;
}

int find_optimal_prunes(Dataset& ds) {
    auto [train_ds, val_ds] = train_test_split_ds(ds, 0.75);

    DecisionTree explore_dt(train_ds, 9999999, 20);

    int curr_prunes = 0;
    int best_prune_count = 0;
    double best_mse = 999999.0;

    std::cout << "Starting Pruning Sequence...\n";

    while (true) {
        double current_mse = calculate_mse(explore_dt, val_ds);
        if (current_mse < best_mse) {
            best_mse = current_mse;
            best_prune_count = curr_prunes;
        }

        if (!explore_dt.prune_one_branch()) {
            break;
        }
        curr_prunes++;
    }

    std::cout << "Optimal state found after " << best_prune_count << " prunes / "<< curr_prunes << " (Lowest Val MSE: " << best_mse << ")\n";
    return best_prune_count;
}

double get_avg_target(Dataset& ds) {
    if (ds.num_rows == 0) {
        std::cout << "Empty dataset!\n";
        return std::nan(0);
    }

    double total = 0;
    for(size_t i = 0; i < ds.num_rows; i++) {
        total += ds.get_target(i);
    }

    return total / ds.num_rows;
}