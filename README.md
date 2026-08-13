# Gradient Boosted Decision Trees

GBDT implementation in C++

## Regression Trees

![tree.webp](tree.webp)


First step would be to implement a **regression tree**. This tree works by partitioning the dataset and choosing a 'best split' - a combination of a feature and threshold. After this, we choose again the next 'best split' recursively on the new 2 partitions.


Recursively partition the tree like this until we reach `max_depth`. After that we close the node, mark it as leaf and this will hold the average of the rows' target values.


## Gradient Boosting

## Project structure

```text
├── inc/
│   ├── Dataset.hpp        # Dataset structure and CSV parsing definitions
│   └── DecisionTree.hpp   # Node structure and decision tree
├── src/
│   ├── Dataset.cpp        # Implementation of data handling and display
│   ├── DecisionTree.cpp   # Implementation of the decision tree logic
│   └── main.cpp           # Main entry point
├── obj/                   # Compiled object files (generated during build)
└── Makefile               # Build automation
```