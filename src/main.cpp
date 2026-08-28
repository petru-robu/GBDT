#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../inc/Dataset.hpp"
#include "../inc/DecisionTree.hpp"
#include "../inc/DecisionTreeUtils.hpp"
#include "../inc/GradientBoostedTree.hpp"

double calculate_mse(Dataset& ds, GradientBoostedTree& model) {
    if (ds.num_rows == 0) {
        std::cerr << "Cannot calculate MSE on an empty dataset!\n";
        return 0.0; 
    }
    std::vector<double> predictions = model.predict_all(ds);
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < ds.num_rows; i++) {
        double true_val = ds.get_target(i);
        double pred_val = predictions[i];
        double error = true_val - pred_val;
        sum_squared_error += (error * error);
    }
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
    std::cout << "Loading datasets...\n";

    std::vector<std::string> cols_to_drop = {
        "Id", "MSZoning", "Street", "Alley", "LotShape", "LandContour", 
        "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", 
        "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", 
        "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", 
        "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
        "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", 
        "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", 
        "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", 
        "MiscFeature", "SaleType", "SaleCondition"
    };

    Dataset train_data = load_csv("data/train.csv", true, -1, cols_to_drop);

    Dataset test_data = load_csv("data/test.csv", true, -2, cols_to_drop);

    // Drop non-numeric columns from both datasets
    for (const auto& col : cols_to_drop) {
        train_data.drop_column(col);
        test_data.drop_column(col); 
    }

    // Split the training data to evaluate performance locally
    std::cout << "========================================\n";
    std::cout << "Evaluating GBDT on 80/20 split...\n";
    std::cout << "========================================\n";
    auto [train_split, val_split] = train_test_split_ds(train_data, 0.8);
    
    GradientBoostedTree gbdt_eval(train_split, 4, 5, 150, 0.1); 
    std::cout << "Validation MSE: " << calculate_mse(val_split, gbdt_eval) << "\n\n";

    std::cout << "Training final GBDT on train.csv...\n";
    GradientBoostedTree gbdt_final(train_data, 4, 5, 150, 0.1);

    // Predict on the test set
    std::vector<double> test_predictions = gbdt_final.predict_all(test_data);

    // Generate submission.csv
    std::ofstream out("data/my_submission.csv");
    out << "Id,SalePrice\n";
    
    // Kaggle housing starts always at 1461
    int current_id = 1461;
    for (size_t i = 0; i < test_predictions.size(); i++) {
        out << current_id << "," << test_predictions[i] << "\n";
        current_id++;
    }
    out.close();

    std::cout << "Saved CSV!\n";
    return 0;
}