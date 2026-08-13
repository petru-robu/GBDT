#include "../inc/Dataset.hpp"
#include "../inc/DecisionTree.hpp"
#include <iostream>

int main() {    
    Dataset ds = load_csv("data/data.csv", true);
    ds.drop_column("id");
    ds.display();
    
    DecisionTree dt(ds, 4);

    return 0;
}