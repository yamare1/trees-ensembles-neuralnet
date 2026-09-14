# Income Classification: Tree Ensembles vs. Neural Networks

Comparing tree-based ensembles against a feed-forward neural network on tabular
data, using U.S. census records to predict whether annual income exceeds $50,000.

## Problem

The Adult dataset (UCI) contains census records with 14 mixed categorical and
continuous features — age, education, occupation, hours worked, capital gains,
and others. The task is binary: does this person earn more than $50k?

The dataset ships with a predefined train/test split, which this project uses
rather than resplitting. Records with missing values (encoded as `?`) are
dropped. Categorical features are one-hot encoded; continuous features are
standardized.

## Results

| Model | Test accuracy | Tuning time |
|---|---|---|
| Majority class baseline | 75.43% |  |
| Neural network (MLP) | 85.05% | 172.6s |
| Decision tree | 85.37% | 0.4s |
| AdaBoost | 85.37% | 7.3s |
| Random forest | 85.78% | 3.3s |
| **Gradient boosting** | **87.07%** | 13.3s |

All models beat the majority-class baseline, but the margins are modest:
+11.6pp for gradient boosting, +9.6pp for the neural network. Accuracy figures
in the mid-80s sound strong until measured against a floor of 75.4%.

Best configuration: gradient boosting, tuned via randomized search over
`n_estimators`, `max_depth`, and `learning_rate`.

## What the comparison shows

**Gradient boosting wins, and the neural network finishes last.** This
reproduces the standard result that tree ensembles outperform neural networks
on tabular data with mixed feature types. The gap is not large in absolute
terms — about two percentage points — but it comes with a 13× difference in
tuning time, and a single decision tree tuned in 0.4 seconds matches the MLP
that took nearly three minutes.

**Four of the five models are within noise of each other.** The random
initialization test in notebook 03 fit the same MLP configuration under five
different seeds and produced validation accuracies spanning 0.8480 to 0.8536 —
a 0.56 percentage point range from seed alone. The decision tree (85.37%),
AdaBoost (85.37%), and neural network (85.05%) all fall inside that band.
Only gradient boosting separates cleanly from the group.

This matters for how the table above should be read: ranking models by a single
accuracy figure implies a precision the experiment does not support. The
defensible claim is that gradient boosting beats the rest and the rest are
roughly equivalent.

**The neural network overfits as it gets larger.** From the architecture sweep:

| Hidden layers | Train acc | Val acc |
|---|---|---|
| (50,) | 0.8947 | 0.8417 |
| (100,) | 0.9139 | 0.8326 |
| (200,) | 0.9310 | 0.8207 |
| (50, 50) | 0.9290 | 0.8248 |
| (100, 50) | 0.9557 | 0.8197 |
| (100, 100) | 0.9674 | 0.8137 |

Training accuracy rises monotonically with capacity while validation accuracy
falls monotonically. The smallest network tested generalizes best. With roughly
30,000 training rows and around 100 one-hot encoded features, there is not
enough data to support the larger architectures.

## Method

**Notebook 01 — preprocessing.** Load, drop missing values, encode the label,
build a `ColumnTransformer` that scales numerics and one-hot encodes
categoricals. An 80/20 train/validation split is carved from the training file;
the provided test file is held out.

**Notebook 02 — tree models.** Hyperparameter curves for each model family
(tree depth, forest size, boosting learning rate) plotting train against
validation error, then randomized search over each model's parameter grid.

**Notebook 03 — neural networks.** Sequential search over architecture,
activation, solver, and learning rate, followed by a seed-stability test and a
final randomized search. Best configuration: single hidden layer of 100 units,
ReLU, SGD, learning rate 0.004, alpha 0.00094.

**Notebook 04 — comparison.** Side-by-side accuracy and tuning time.

## Limitations

- **Accuracy is the wrong primary metric** given the class imbalance. The
  minority class (>$50k) is the one of interest, and accuracy hides
  minority-class performance. Precision, recall, and F1 on the positive class
  would be more informative, and are not reported here.
- **The MLPs did not converge.** Every neural network fit hit the 300-iteration
  cap with a `ConvergenceWarning`. The reported neural network numbers are
  therefore a lower bound; longer training might close some of the gap.
- **Listwise deletion of missing values** drops around 7% of records rather than
  imputing. This assumes missingness is unrelated to income, which is not tested.
- **Demographic features** (race, sex, native-country) are included as
  predictors. A model trained on them reproduces the disparities present in
  1994 census data. No fairness auditing was performed.
- **The dataset is from the 1994 census.** Results describe that population.

## Structure

```
01_data_exploration.ipynb    preprocessing, encoding, train/val/test split
02_tree_based_models.ipynb   decision tree, random forest, AdaBoost, gradient boosting
03_neural_networks.ipynb     architecture search, seed stability, final tuning
04_model_comparison.ipynb    accuracy and runtime comparison
adult.data / adult.test      raw UCI data
adult.names                  feature documentation
```

Notebooks run in numerical order and pass state through pickle files in
`data/processed/`.

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter notebook
```
