#include "../inc/Dataset.hpp"
#include "../inc/DecisionTree.hpp"
#include "../inc/DecisionTreeUtils.hpp"
#include "../inc/GradientBoostedTree.hpp"

double calculate_mse(Dataset& ds, GradientBoostedTree& model) {
    // 1. Safety check for empty dataset
    if (ds.num_rows == 0) {
        std::cerr << "Cannot calculate MSE on an empty dataset!\n";
        return 0.0; 
    }

    // 2. Generate predictions for the entire dataset using the model
    std::vector<double> predictions = model.predict_all(ds);
    
    double sum_squared_error = 0.0;

    // 3. Loop through every row to calculate the squared errors
    for (size_t i = 0; i < ds.num_rows; i++) {
        double true_val = ds.get_target(i);
        double pred_val = predictions[i];
        
        double error = true_val - pred_val;
        sum_squared_error += (error * error);
    }

    // 4. Return the mean of the squared errors
    return sum_squared_error / ds.num_rows;
}

double calculate_mse(Dataset& ds, DecisionTree& tree) {
    if (ds.num_rows == 0) return 0.0; 

    double sum_squared_error = 0.0;
    for (size_t i = 0; i < ds.num_rows; i++) {
        double true_val = ds.get_target(i);
        double pred_val = tree.predict(ds, i);
        
        double error = true_val - pred_val;
        sum_squared_error += (error * error);
    }
    return sum_squared_error / ds.num_rows;
}

int main() {
    // 1. Load and prepare the data
    Dataset full_data = load_csv("data/data.csv", true);
    full_data.drop_column("id"); // Don't forget to drop ID!
    
    // Split the data (80% train, 20% test)
    auto [train_ds, test_ds] = train_test_split_ds(full_data, 0.8);

    std::cout << "========================================\n";
    std::cout << "Training Standard Decision Tree...\n";
    std::cout << "========================================\n";
    // Using deeper depth and larger min_samples as standard trees easily overfit
    DecisionTree dt(train_ds, 15, 20); 
    
    double dt_train_mse = calculate_mse(train_ds, dt);
    double dt_test_mse = calculate_mse(test_ds, dt);
    
    std::cout << "Standard Tree Train MSE: " << dt_train_mse << "\n";
    std::cout << "Standard Tree Test MSE:  " << dt_test_mse << "\n\n";

    std::cout << "========================================\n";
    std::cout << "Training Gradient Boosted Tree...\n";
    std::cout << "========================================\n";
    // Using shallow depth (weak learners) but 100 iterations
    GradientBoostedTree gbdt(train_ds, 3, 5, 100, 0.1); 
    
    double gbdt_train_mse = calculate_mse(train_ds, gbdt);
    double gbdt_test_mse = calculate_mse(test_ds, gbdt);
    
    std::cout << "GBDT Train MSE: " << gbdt_train_mse << "\n";
    std::cout << "GBDT Test MSE:  " << gbdt_test_mse << "\n\n";

    // 3. Final Comparison Output
    std::cout << "========================================\n";
    std::cout << "Final Comparison (Test Set):\n";
    std::cout << "========================================\n";
    if (gbdt_test_mse < dt_test_mse) {
        std::cout << "GBDT performed better by " << (dt_test_mse - gbdt_test_mse) << " MSE points!\n";
    } else {
        std::cout << "Standard Tree performed better by " << (gbdt_test_mse - dt_test_mse) << " MSE points.\n";
    }

    return 0;
}