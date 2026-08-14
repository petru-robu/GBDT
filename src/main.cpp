#include "../inc/Dataset.hpp"
#include "../inc/DecisionTree.hpp"
#include "../inc/DecisionTreeUtils.hpp"


int main() {    
    Dataset ds = load_csv("data/data.csv", true);
    ds.drop_column("id");
    DecisionTree tree(ds, 15, 20);
    evaluate_split_dt(ds, tree);

    // prune!
    std::cout << "\nOptimizing Tree: \n";
    int optimal_cuts = find_optimal_prunes(ds);
    DecisionTree final_tree(ds, 9999999, 2); 
    for (int i = 0; i < optimal_cuts; i++) {
        final_tree.prune_one_branch();
    }
    
    evaluate_split_dt(ds, final_tree);

    return 0;
}