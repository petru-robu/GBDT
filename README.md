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
    - because $\sum x_i^2$ is constant, we just maximize $\frac{(\sum x_i)^2}{n} = \text{Average} \cdot \text{Sum}$
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

Gradient boosting is a technique to optimize decision trees even further. On a normal decision tree we might overfit, in gradient boosting we consider an ensamble of tree. We fit the tree on the residuals and then predict, we compute new residuals and fit again on this - until we reach a fixed threshold. T

- This way we are actually improving over time by sequentially minimizing our errors. Because each new tree specifically targets the mistakes of the combined ensemble that came before it, the model acts as a self correcting system.

- Learning rate: We don't add the raw prediction of the trees to the total model, we multiply with a learning rate (ex. 0.01) before adding it to take small steps.

- Final prediciton: After fitting, to predict using our tree ensamble we just sum the scaled predictions of every tree.

## Testing

What better way to test the model's performance than with a regression Kaggle contest - [House Prices, Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- We are throwing away ~half of the dataset, because the my Dataset class support only numeric features (encoding and other ML techniques are not the focus of this project, so I didn't bother implementing them)

- The Kaggle result is **0.14054**, which is not so bad on this contest, so my tree is actually for real


## Project structure

```text
├── inc/
│   ├── Dataset.hpp             # Dataset structure and CSV parsing definitions
│   ├── DecisionTreeUtils.hpp   # Other utilites
│   ├── GradientBoostedTree.hpp # GBDT structure
│   └── DecisionTree.hpp        # Node structure and decision tree
├── src/
│   ├── Dataset.cpp             # Implementation of data handling and display
│   ├── DecisionTreeUtils.cpp   # Other utilites
│   ├── GradientBoostedTree.cpp # Implementation of the gbdt logic
│   ├── DecisionTree.cpp        # Implementation of the decision tree logic
│   └── main.cpp                # Main entry point
├── obj/                        # Compiled object files (generated during build)
├── scripts/generate_ds.py      # A helper script to generate a very simple dataset for testing
├── data                        # data folder with csv's (gitignored data) - just plug in Kaggle Housing data if you wish to test
└── Makefile                    # Build automation
```

## Running instructions

Just run the makefile and the the executable:

```bash
make
./decision_tree
```


## References
- *G. James, D. Witten, T. Hastie, R. Tibshirani, and J. Taylor*, An Introduction to Statistical Learning: With Applications in Python. *Springer*, 2023.

