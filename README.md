# Gradient Boosted Decision Trees

Gradient Boosted Decision Trees implementation in C++.

## Regression Trees

![tree.webp](tree.webp)

First step would be to implement a **regression tree**. This tree works by partitioning the dataset and choosing a 'best split' - a combination of a feature and threshold. After this, we choose again the next 'best split' recursively on the 2 new partitions.

### Splitting the dataset

The 'best split' is determined like this:
- For every feature, calculate threshold:
    - sort region by feature ascending, move threshold from left to right one by one to form two right and left subregions
    - on the left and right subregions, compute **RSS** (Residual Sum of Squares) sum of both bukets and minimize it: $$\sum (x_i - \bar{x})^2 = \sum x_i^2 - \frac{(\sum x_i)^2}{n}$$
    - because $ \sum x_i^2 $ is constant, we just maximize $\frac{(\sum x_i)^2}{n} = \text{Average} \cdot \text{Sum}$
- Keep feature + threshold with lowest RSS - that's the best split at this moment and a node in the tree

Recursively partition the tree like this until we reach `max_depth`. After that we close the node, mark it as leaf and this will hold the average of the rows' target values.

### Pruning

Pruning is a DT optimization. Because the tree is likely to overfit and tuning `max_depth` and `minimum_region_size` paramaters is not the best option, we can use **pruning**. This means we build a big tree (complete - every leaf = 1 row of our dataset) and then find what branches we can remove to minimize it's validation MSE. 

For this we keep track in each node of the tree of the errors of

    - 1) RSS if it was leaf right now: `actual value` - `mean value of the bucket`

    - 2) RSS of the subtrees (left + right)

And then if we compute (`1)` - `2)`) / `number_of_leaves_in_subtree - 1` we get how much less error we have if we make the split vs just if we close there and make the leaf - that's the **weak_link**

Then **prune** (remove weakest_link and recalculate tree errors) one-by-one until tree collapses in a single leaf, but record along the way the MSEs. Pick the tree with best MSE.


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