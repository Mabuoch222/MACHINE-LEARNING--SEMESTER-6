# Machine Learning: Numerical Topics and Overfitting/Underfitting Guide

## Table of Contents
- [Numerical Topics in Machine Learning](#numerical-topics-in-machine-learning)
  - [1. Linear Regression](#1-linear-regression)
  - [2. Logistic Regression](#2-logistic-regression)
  - [3. Gradient Descent](#3-gradient-descent)
  - [4. Cost Functions and Loss Functions](#4-cost-functions-and-loss-functions)
  - [5. Regularization](#5-regularization)
  - [6. Evaluation Metrics](#6-evaluation-metrics)
  - [7. Cross-Validation](#7-cross-validation)
  - [8. Confusion Matrix](#8-confusion-matrix)
  - [9. Bias-Variance Tradeoff](#9-bias-variance-tradeoff)
  - [10. Principal Component Analysis (PCA)](#10-principal-component-analysis-pca)
  - [11. K-Means Clustering](#11-k-means-clustering)
  - [12. Support Vector Machines (SVM)](#12-support-vector-machines-svm)
  - [13. Decision Trees](#13-decision-trees)
  - [14. Neural Networks](#14-neural-networks)
  - [15. Learning Rate and Optimization](#15-learning-rate-and-optimization)
- [Overfitting vs Underfitting](#overfitting-vs-underfitting)
- [Techniques to Mitigate Problems](#techniques-to-mitigate-problems)

---

## Numerical Topics in Machine Learning

### 1. Linear Regression

Linear regression models the relationship between independent variables (features) and a dependent variable (target) using a linear equation.

#### Formula

**Simple Linear Regression:**
```
y = β₀ + β₁x + ε

Where:
- y = predicted value (dependent variable)
- x = independent variable (feature)
- β₀ = intercept (bias term)
- β₁ = slope (weight/coefficient)
- ε = error term
```

**Multiple Linear Regression:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Matrix form: y = Xβ + ε
```

#### Cost Function (Mean Squared Error)

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Where:
- n = number of samples
- yᵢ = actual value
- ŷᵢ = predicted value
```

#### Normal Equation (Closed-form Solution)

```
β = (XᵀX)⁻¹Xᵀy

Where:
- X = feature matrix
- y = target vector
- β = coefficient vector
```

#### Numerical Example

**Problem:** Predict house price based on size

| Size (sq ft) | Price ($1000) |
|--------------|---------------|
| 1000         | 200           |
| 1500         | 250           |
| 2000         | 300           |
| 2500         | 350           |

**Solution:**

1. Calculate means: x̄ = 1750, ȳ = 275

2. Calculate slope (β₁):
```
β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²
β₁ = 0.1
```

3. Calculate intercept (β₀):
```
β₀ = ȳ - β₁x̄
β₀ = 275 - (0.1 × 1750) = 100
```

4. Final equation:
```
Price = 100 + 0.1 × Size
```

5. Prediction for 1800 sq ft:
```
Price = 100 + 0.1 × 1800 = 280 ($280,000)
```

---

### 2. Logistic Regression

Logistic regression is used for binary classification problems, predicting probabilities between 0 and 1.

#### Sigmoid Function

```
σ(z) = 1 / (1 + e⁻ᶻ)

Where: z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

#### Hypothesis Function

```
h(x) = σ(βᵀx) = 1 / (1 + e⁻ᵝᵀˣ)
```

#### Cost Function (Log Loss/Binary Cross-Entropy)

```
J(β) = -(1/n) Σ[yᵢ log(h(xᵢ)) + (1-yᵢ) log(1-h(xᵢ))]

Where:
- yᵢ ∈ {0, 1}
- h(xᵢ) = predicted probability
```

#### Decision Boundary

```
Predict y = 1 if h(x) ≥ 0.5 (i.e., z ≥ 0)
Predict y = 0 if h(x) < 0.5 (i.e., z < 0)
```

#### Numerical Example

**Problem:** Predict if a student passes (1) or fails (0) based on study hours

| Hours Studied | Pass (1) / Fail (0) |
|---------------|---------------------|
| 1             | 0                   |
| 2             | 0                   |
| 3             | 0                   |
| 4             | 1                   |
| 5             | 1                   |
| 6             | 1                   |

**Solution:**

Assume trained model: z = -5 + 1.5x

For x = 3.5 hours:
```
z = -5 + 1.5(3.5) = -0.25
σ(z) = 1/(1 + e⁰·²⁵) = 1/(1 + 1.284) = 0.438

Since 0.438 < 0.5, predict Fail (0)
```

For x = 5 hours:
```
z = -5 + 1.5(5) = 2.5
σ(z) = 1/(1 + e⁻²·⁵) = 1/(1 + 0.082) = 0.924

Since 0.924 ≥ 0.5, predict Pass (1)
```

---

### 3. Gradient Descent

Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating parameters.

#### Algorithm

```
Repeat until convergence:
    β := β - α ∂J(β)/∂β

Where:
- β = parameters (weights)
- α = learning rate
- ∂J(β)/∂β = gradient (partial derivative)
```

#### Types of Gradient Descent

**1. Batch Gradient Descent**
```
βⱼ := βⱼ - α (1/n) Σ(h(xᵢ) - yᵢ)xᵢⱼ

Uses entire dataset for each update
```

**2. Stochastic Gradient Descent (SGD)**
```
βⱼ := βⱼ - α (h(xᵢ) - yᵢ)xᵢⱼ

Updates parameters for each training example
```

**3. Mini-Batch Gradient Descent**
```
Uses small batches (e.g., 32, 64, 128 samples)
Balance between batch and stochastic
```

#### Numerical Example

**Problem:** Minimize J(β) = (β - 3)² using gradient descent

Given: α = 0.1, initial β₀ = 0

**Solution:**

Gradient: dJ/dβ = 2(β - 3)

| Iteration | β     | J(β)  | Gradient | Update        |
|-----------|-------|-------|----------|---------------|
| 0         | 0.0   | 9.0   | -6.0     | β := 0 + 0.6  |
| 1         | 0.6   | 5.76  | -4.8     | β := 0.6 + 0.48|
| 2         | 1.08  | 3.69  | -3.84    | β := 1.08 + 0.384|
| 3         | 1.464 | 2.36  | -3.072   | β := 1.464 + 0.3072|
| 4         | 1.771 | 1.51  | -2.458   | ...           |
| ...       | ...   | ...   | ...      | ...           |
| ∞         | 3.0   | 0.0   | 0.0      | Converged     |

---

### 4. Cost Functions and Loss Functions

Cost functions measure how well the model performs. Different problems require different cost functions.

#### Mean Squared Error (MSE) - Regression

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

#### Mean Absolute Error (MAE) - Regression

```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

#### Root Mean Squared Error (RMSE) - Regression

```
RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]
```

#### Binary Cross-Entropy - Binary Classification

```
BCE = -(1/n) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

#### Categorical Cross-Entropy - Multi-class Classification

```
CCE = -(1/n) Σ Σ yᵢⱼ log(ŷᵢⱼ)
      i   j

Where:
- i = samples
- j = classes
```

#### Hinge Loss - SVM

```
L = max(0, 1 - yᵢ · ŷᵢ)
```

#### Numerical Example

**Problem:** Calculate MSE and MAE

| Actual (y) | Predicted (ŷ) | Error | Squared Error | Absolute Error |
|------------|---------------|-------|---------------|----------------|
| 10         | 12            | 2     | 4             | 2              |
| 15         | 13            | -2    | 4             | 2              |
| 20         | 22            | 2     | 4             | 2              |
| 25         | 24            | -1    | 1             | 1              |

**Solution:**

```
MSE = (4 + 4 + 4 + 1) / 4 = 13/4 = 3.25

MAE = (2 + 2 + 2 + 1) / 4 = 7/4 = 1.75

RMSE = √3.25 = 1.80
```

---

### 5. Regularization

Regularization prevents overfitting by adding a penalty term to the cost function.

#### L1 Regularization (Lasso)

```
J(β) = MSE + λ Σ|βⱼ|

Where:
- λ = regularization parameter
- Promotes sparsity (some weights become exactly 0)
```

#### L2 Regularization (Ridge)

```
J(β) = MSE + λ Σβⱼ²

Where:
- λ = regularization parameter
- Shrinks weights but doesn't make them zero
```

#### Elastic Net

```
J(β) = MSE + λ₁ Σ|βⱼ| + λ₂ Σβⱼ²

Combination of L1 and L2
```

#### Numerical Example

**Problem:** Ridge regression with regularization

Original cost: MSE = 10
Weights: β₁ = 3, β₂ = 4, β₃ = 2
λ = 0.1

**Solution:**

```
Penalty = λ(β₁² + β₂² + β₃²)
        = 0.1(9 + 16 + 4)
        = 0.1(29)
        = 2.9

Total Cost = MSE + Penalty
          = 10 + 2.9
          = 12.9
```

---

### 6. Evaluation Metrics

#### Regression Metrics

**R² (R-Squared / Coefficient of Determination)**
```
R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

Range: (-∞, 1]
- R² = 1: Perfect fit
- R² = 0: Model as good as mean
- R² < 0: Model worse than mean
```

**Adjusted R²**
```
Adjusted R² = 1 - [(1 - R²)(n - 1)/(n - p - 1)]

Where:
- n = number of samples
- p = number of predictors
```

#### Classification Metrics

**Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives
```

**Precision**
```
Precision = TP / (TP + FP)

Of all positive predictions, how many were correct?
```

**Recall (Sensitivity/True Positive Rate)**
```
Recall = TP / (TP + FN)

Of all actual positives, how many did we catch?
```

**F1-Score**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Harmonic mean of precision and recall
```

**Specificity (True Negative Rate)**
```
Specificity = TN / (TN + FP)
```

#### Numerical Example

**Problem:** Calculate metrics for a binary classifier

| Actual | Predicted |
|--------|-----------|
| 1      | 1         | TP
| 1      | 0         | FN
| 0      | 1         | FP
| 0      | 0         | TN
| 1      | 1         | TP
| 0      | 0         | TN
| 1      | 1         | TP
| 0      | 1         | FP

Count: TP=3, TN=2, FP=2, FN=1

**Solution:**

```
Accuracy = (3 + 2) / (3 + 2 + 2 + 1) = 5/8 = 0.625 = 62.5%

Precision = 3 / (3 + 2) = 3/5 = 0.6 = 60%

Recall = 3 / (3 + 1) = 3/4 = 0.75 = 75%

F1-Score = 2 × (0.6 × 0.75) / (0.6 + 0.75)
         = 2 × 0.45 / 1.35
         = 0.667 = 66.7%

Specificity = 2 / (2 + 2) = 2/4 = 0.5 = 50%
```

---

### 7. Cross-Validation

Cross-validation assesses model performance by training and testing on different data subsets.

#### K-Fold Cross-Validation

```
1. Split data into K equal folds
2. For each fold i (i = 1 to K):
   - Use fold i as test set
   - Use remaining K-1 folds as training set
   - Train model and evaluate
3. Average performance across all K folds
```

#### Formula

```
CV Score = (1/K) Σ Score_i

Where Score_i is the performance on fold i
```

#### Numerical Example

**Problem:** 5-Fold Cross-Validation with accuracy scores

| Fold | Training Folds | Test Fold | Accuracy |
|------|----------------|-----------|----------|
| 1    | 2,3,4,5        | 1         | 0.85     |
| 2    | 1,3,4,5        | 2         | 0.88     |
| 3    | 1,2,4,5        | 3         | 0.82     |
| 4    | 1,2,3,5        | 4         | 0.87     |
| 5    | 1,2,3,4        | 5         | 0.86     |

**Solution:**

```
Average CV Score = (0.85 + 0.88 + 0.82 + 0.87 + 0.86) / 5
                 = 4.28 / 5
                 = 0.856 = 85.6%

Standard Deviation = 0.021
```

---

### 8. Confusion Matrix

A confusion matrix visualizes the performance of a classification model.

#### Structure

```
                Predicted
              Pos      Neg
Actual  Pos   TP       FN
        Neg   FP       TN
```

#### Multi-class Confusion Matrix

```
              Predicted
           Class A  Class B  Class C
Actual A      15       2        3
       B       1      18        1
       C       2       1       17
```

#### Numerical Example

**Problem:** Binary classification results

Predictions: [1, 0, 1, 1, 0, 0, 1, 0]
Actual:      [1, 0, 1, 0, 1, 0, 1, 1]

**Solution:**

Count:
- TP (predicted 1, actual 1): 3
- TN (predicted 0, actual 0): 2
- FP (predicted 1, actual 0): 1
- FN (predicted 0, actual 1): 2

```
Confusion Matrix:
              Predicted
           Pos(1)  Neg(0)
Actual 1     3       2
       0     1       2

From this:
Accuracy = (3+2)/8 = 62.5%
Precision = 3/(3+1) = 75%
Recall = 3/(3+2) = 60%
```

---

### 9. Bias-Variance Tradeoff

The bias-variance tradeoff is fundamental to understanding model performance.

#### Components

**Total Error**
```
Error = Bias² + Variance + Irreducible Error

Where:
- Bias² = (f̂(x) - f(x))²
- Variance = E[(f̂(x) - E[f̂(x)])²]
- Irreducible Error = σ² (inherent noise)
```

#### Definitions

**Bias:**
```
Bias = E[f̂(x)] - f(x)

- Error from wrong assumptions
- High bias → underfitting
- Measures how far off predictions are on average
```

**Variance:**
```
Variance = E[(f̂(x) - E[f̂(x)])²]

- Error from sensitivity to training data
- High variance → overfitting
- Measures how much predictions vary for different training sets
```

#### Numerical Example

**Problem:** Compare models on different datasets

Model A (Simple - High Bias, Low Variance):
- Dataset 1: MSE = 25
- Dataset 2: MSE = 24
- Dataset 3: MSE = 26
- Average MSE = 25, Std Dev = 0.82

Model B (Complex - Low Bias, High Variance):
- Dataset 1: MSE = 10
- Dataset 2: MSE = 30
- Dataset 3: MSE = 15
- Average MSE = 18.3, Std Dev = 8.50

**Analysis:**
- Model A: Consistent but higher error (high bias, low variance)
- Model B: Lower average error but inconsistent (low bias, high variance)

---

### 10. Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that transforms data into principal components.

#### Steps

1. **Standardize the data**
```
z = (x - μ) / σ
```

2. **Compute covariance matrix**
```
Cov(X) = (1/n) XᵀX
```

3. **Calculate eigenvalues and eigenvectors**
```
Cov(X)v = λv

Where:
- λ = eigenvalue
- v = eigenvector
```

4. **Select top k eigenvectors**
```
Sort by eigenvalues (largest to smallest)
Keep top k components
```

5. **Transform data**
```
Z = X · V_k

Where V_k contains k eigenvectors
```

#### Explained Variance

```
Explained Variance Ratio = λᵢ / Σλⱼ

Cumulative Explained Variance = Σ(λᵢ / Σλⱼ)
```

#### Numerical Example

**Problem:** Reduce 2D data to 1D using PCA

Data:
```
X = [[2, 3],
     [3, 4],
     [4, 5],
     [5, 6]]
```

**Solution:**

1. Mean: μ = [3.5, 4.5]

2. Centered data:
```
X_centered = [[-1.5, -1.5],
              [-0.5, -0.5],
              [ 0.5,  0.5],
              [ 1.5,  1.5]]
```

3. Covariance matrix:
```
Cov = [[1.67, 1.67],
       [1.67, 1.67]]
```

4. Eigenvalues: λ₁ = 3.33, λ₂ = 0
   Eigenvectors: v₁ = [0.707, 0.707], v₂ = [-0.707, 0.707]

5. Transform using v₁:
```
Z = [[-2.12],
     [-0.71],
     [ 0.71],
     [ 2.12]]
```

6. Explained variance: 3.33/3.33 = 100%

---

### 11. K-Means Clustering

K-Means is an unsupervised algorithm that partitions data into K clusters.

#### Algorithm

```
1. Initialize K centroids randomly
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids as mean of assigned points
3. Stop when centroids don't change
```

#### Distance Metric (Euclidean)

```
d(x, c) = √(Σ(xᵢ - cᵢ)²)

Where:
- x = data point
- c = centroid
```

#### Cost Function (Inertia)

```
J = Σ Σ ||x - μₖ||²
    k x∈Cₖ

Where:
- Cₖ = cluster k
- μₖ = centroid of cluster k
```

#### Elbow Method

```
Plot inertia vs. number of clusters (K)
Choose K at the "elbow" point
```

#### Numerical Example

**Problem:** Cluster 6 points into 2 clusters

Data: [2, 3, 8, 9, 10, 12]

**Solution:**

Initial centroids: c₁ = 2, c₂ = 12

**Iteration 1:**

Assign to nearest centroid:
- Point 2: |2-2|=0, |2-12|=10 → Cluster 1
- Point 3: |3-2|=1, |3-12|=9 → Cluster 1
- Point 8: |8-2|=6, |8-12|=4 → Cluster 2
- Point 9: |9-2|=7, |9-12|=3 → Cluster 2
- Point 10: |10-2|=8, |10-12|=2 → Cluster 2
- Point 12: |12-2|=10, |12-12|=0 → Cluster 2

Clusters: C₁ = {2, 3}, C₂ = {8, 9, 10, 12}

Update centroids:
- c₁ = (2+3)/2 = 2.5
- c₂ = (8+9+10+12)/4 = 9.75

**Iteration 2:**

Assign to nearest centroid:
- Point 2: |2-2.5|=0.5, |2-9.75|=7.75 → Cluster 1
- Point 3: |3-2.5|=0.5, |3-9.75|=6.75 → Cluster 1
- Point 8: |8-2.5|=5.5, |8-9.75|=1.75 → Cluster 2
- Point 9: |9-2.5|=6.5, |9-9.75|=0.75 → Cluster 2
- Point 10: |10-2.5|=7.5, |10-9.75|=0.25 → Cluster 2
- Point 12: |12-2.5|=9.5, |12-9.75|=2.25 → Cluster 2

Clusters: C₁ = {2, 3}, C₂ = {8, 9, 10, 12}

Centroids unchanged → **Converged**

Final clusters: C₁ = {2, 3}, C₂ = {8, 9, 10, 12}

---

### 12. Support Vector Machines (SVM)

SVM finds the optimal hyperplane that maximizes the margin between classes.

#### Linear SVM

**Decision Boundary:**
```
wᵀx + b = 0

Where:
- w = weight vector (normal to hyperplane)
- b = bias
- x = input features
```

**Margin:**
```
Margin = 2/||w||

Goal: Maximize margin = Minimize ||w||
```

**Optimization Problem:**
```
Minimize: (1/2)||w||²

Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 for all i

Where:
- yᵢ ∈ {-1, +1}
```

#### Soft Margin SVM (with slack variables)

```
Minimize: (1/2)||w||² + C Σξᵢ

Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0

Where:
- C = regularization parameter
- ξᵢ = slack variable (allows misclassification)
```

#### Kernel Functions

**Linear Kernel:**
```
K(x, x') = xᵀx'
```

**Polynomial Kernel:**
```
K(x, x') = (xᵀx' + c)ᵈ
```

**RBF (Gaussian) Kernel:**
```
K(x, x') = exp(-γ||x - x'||²)
```

#### Numerical Example

**Problem:** Find SVM decision boundary for 2D data

Class +1: (3, 3), (4, 4)
Class -1: (1, 1), (2, 2)

Assume linear boundary: w₁x₁ + w₂x₂ + b = 0

**Solution (simplified):**

For perfectly separable data with support vectors:
- (2, 2) for class -1
- (3, 3) for class +1

Assuming w = [1, 1], we need to find b:

For (2, 2): -1(1×2 + 1×2 + b) ≥ 1 → b ≤ -5
For (3, 3): +1(1×3 + 1×3 + b) ≥ 1 → b ≥ -5

Decision boundary: x₁ + x₂ - 5 = 0
or equivalently: x₂ = -x₁ + 5

Margin width: 2/√(1² + 1²) = 2/√2 = √2 ≈ 1.41

---

### 13. Decision Trees

Decision trees make predictions by learning decision rules from features.

#### Information Gain

**Entropy:**
```
H(S) = -Σ pᵢ log₂(pᵢ)

Where:
- pᵢ = proportion of class i in set S
```

**Information Gain:**
```
IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) H(Sᵥ)
                  v∈Values(A)

Where:
- A = attribute
- Sᵥ = subset of S where A = v
```

#### Gini Impurity

```
Gini(S) = 1 - Σ pᵢ²

Where:
- pᵢ = proportion of class i
```

#### Numerical Example

**Problem:** Calculate entropy and information gain

Dataset (12 samples):
- 8 Class A (positive)
- 4 Class B (negative)

Feature: Color (Red: 5A, 1B; Blue: 3A, 3B)

**Solution:**

1. Parent entropy:
```
p(A) = 8/12 = 2/3
p(B) = 4/12 = 1/3

H(Parent) = -[2/3 log₂(2/3) + 1/3 log₂(1/3)]
          = -[2/3(-0.585) + 1/3(-1.585)]
          = -[-0.39 - 0.528]
          = 0.918
```

2. Entropy after split:

Red subset (6 samples: 5A, 1B):
```
H(Red) = -[5/6 log₂(5/6) + 1/6 log₂(1/6)]
       = -[5/6(-0.263) + 1/6(-2.585)]
       = 0.650
```

Blue subset (6 samples: 3A, 3B):
```
H(Blue) = -[3/6 log₂(3/6) + 3/6 log₂(3/6)]
        = -[1/2(-1) + 1/2(-1)]
        = 1.0
```

3. Weighted average:
```
H(Split) = (6/12 × 0.650) + (6/12 × 1.0)
         = 0.325 + 0.5
         = 0.825
```

4. Information Gain:
```
IG = H(Parent) - H(Split)
   = 0.918 - 0.825
   = 0.093
```

---

### 14. Neural Networks

Neural networks consist of interconnected neurons organized in layers.

#### Forward Propagation

**Single Neuron:**
```
z = Σ wᵢxᵢ + b
a = σ(z)

Where:
- wᵢ = weights
- xᵢ = inputs
- b = bias
- σ = activation function
- a = output
```

**Layer-wise:**
```
Z[l] = W[l]A[l-1] + b[l]
A[l] = σ(Z[l])

Where:
- l = layer number
- W[l] = weight matrix for layer l
- A[l-1] = activations from previous layer
```

#### Activation Functions

**Sigmoid:**
```
σ(z) = 1/(1 + e⁻ᶻ)
σ'(z) = σ(z)(1 - σ(z))
```

**ReLU:**
```
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0, else 0
```

**Tanh:**
```
tanh(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)
tanh'(z) = 1 - tanh²(z)
```

#### Backpropagation

**Output Layer:**
```
δ[L] = (A[L] - Y) ⊙ σ'(Z[L])

Where:
- ⊙ = element-wise multiplication
- L = output layer
```

**Hidden Layers:**
```
δ[l] = (W[l+1]ᵀδ[l+1]) ⊙ σ'(Z[l])
```

**Gradient Computation:**
```
dW[l] = (1/m) δ[l]A[l-1]ᵀ
db[l] = (1/m) Σ δ[l]

Where m = batch size
```

**Weight Update:**
```
W[l] := W[l] - α dW[l]
b[l] := b[l] - α db[l]

Where α = learning rate
```

#### Numerical Example

**Problem:** Forward pass through a simple network

Network: 2 inputs → 1 hidden neuron → 1 output
Input: x = [0.5, 0.3]
Weights: W₁ = [0.4, 0.6], b₁ = 0.1
         W₂ = [0.8], b₂ = 0.2
Activation: Sigmoid

**Solution:**

1. Hidden layer:
```
z₁ = 0.4(0.5) + 0.6(0.3) + 0.1
   = 0.2 + 0.18 + 0.1
   = 0.48

a₁ = σ(0.48) = 1/(1 + e⁻⁰·⁴⁸) = 0.618
```

2. Output layer:
```
z₂ = 0.8(0.618) + 0.2
   = 0.494 + 0.2
   = 0.694

a₂ = σ(0.694) = 1/(1 + e⁻⁰·⁶⁹⁴) = 0.667
```

Final output: 0.667

---

### 15. Learning Rate and Optimization

The learning rate controls how much parameters are updated during training.

#### Learning Rate

```
θ := θ - α ∇J(θ)

Where:
- α = learning rate
- Too high → divergence
- Too low → slow convergence
```

#### Momentum

```
v := βv + (1-β)∇J(θ)
θ := θ - α v

Where:
- β = momentum coefficient (typically 0.9)
- v = velocity (exponentially weighted average of gradients)
```

#### Adam Optimizer

```
m := β₁m + (1-β₁)∇J(θ)         (first moment)
v := β₂v + (1-β₂)(∇J(θ))²      (second moment)

m̂ := m/(1-β₁ᵗ)                 (bias correction)
v̂ := v/(1-β₂ᵗ)

θ := θ - α m̂/(√v̂ + ε)

Where:
- β₁ = 0.9 (typical)
- β₂ = 0.999 (typical)
- ε = 10⁻⁸ (small constant)
- t = time step
```

#### Learning Rate Schedules

**Step Decay:**
```
α(t) = α₀ × decay^floor(t/step_size)
```

**Exponential Decay:**
```
α(t) = α₀ × e^(-kt)
```

**1/t Decay:**
```
α(t) = α₀/(1 + kt)
```

#### Numerical Example

**Problem:** Compare gradient descent with and without momentum

Gradient at each step: [10, 8, 6, 4]
Learning rate: α = 0.1
Momentum: β = 0.9
Initial: θ = 0, v = 0

**Solution:**

**Without Momentum:**
| Step | Gradient | Update      | θ      |
|------|----------|-------------|--------|
| 1    | 10       | -0.1(10)    | -1.0   |
| 2    | 8        | -0.1(8)     | -1.8   |
| 3    | 6        | -0.1(6)     | -2.4   |
| 4    | 4        | -0.1(4)     | -2.8   |

**With Momentum:**
| Step | Gradient | v                      | Update    | θ      |
|------|----------|------------------------|-----------|--------|
| 1    | 10       | 0.9(0) + 0.1(10) = 1   | -0.1(1)   | -0.1   |
| 2    | 8        | 0.9(1) + 0.1(8) = 1.7  | -0.1(1.7) | -0.27  |
| 3    | 6        | 0.9(1.7) + 0.1(6) = 2.13| -0.1(2.13)| -0.483 |
| 4    | 4        | 0.9(2.13) + 0.1(4) = 2.32| -0.1(2.32)| -0.715|

Momentum accumulates gradients, leading to faster convergence.

---

## Overfitting vs Underfitting

Understanding and addressing overfitting and underfitting is crucial for building effective machine learning models.

### What is Underfitting?

**Definition:** Underfitting occurs when a model is too simple to capture the underlying patterns in the data.

**Characteristics:**
- High training error
- High test error
- Model is too simplistic
- High bias, low variance
- Poor performance on both training and test data

**Indicators:**
```
Training Accuracy: 60%
Test Accuracy: 58%
→ Both low, similar values = Underfitting
```

**Visual Example:**
```
Actual data: Complex curved pattern
Underfit model: Straight line (linear) trying to fit curved data
Result: Systematic errors, misses the pattern
```

**Common Causes:**
- Model is too simple (e.g., linear model for non-linear data)
- Too few features
- Too much regularization
- Insufficient training time
- Poor feature engineering

---

### What is Overfitting?

**Definition:** Overfitting occurs when a model learns the training data too well, including noise and outliers, failing to generalize to new data.

**Characteristics:**
- Very low training error
- High test error
- Model is too complex
- Low bias, high variance
- Excellent on training data, poor on test data

**Indicators:**
```
Training Accuracy: 99%
Test Accuracy: 65%
→ Large gap = Overfitting
```

**Visual Example:**
```
Actual data: Points with some noise
Overfit model: Curve passing through every single point
Result: Memorizes noise, doesn't generalize
```

**Common Causes:**
- Model is too complex (too many parameters)
- Too many features relative to training samples
- Insufficient regularization
- Training for too long
- Not enough training data
- No validation during training

---

### Comparison Table

| Aspect | Underfitting | Good Fit | Overfitting |
|--------|--------------|----------|-------------|
| **Training Error** | High | Low | Very Low |
| **Test Error** | High | Low | High |
| **Model Complexity** | Too Simple | Appropriate | Too Complex |
| **Bias** | High | Balanced | Low |
| **Variance** | Low | Balanced | High |
| **Training Accuracy** | ~60% | ~85% | ~99% |
| **Test Accuracy** | ~58% | ~83% | ~65% |
| **Generalization** | Poor | Good | Poor |
| **Pattern Capture** | Misses patterns | Captures patterns | Captures noise |

---

### Visual Representation

```
Model Complexity vs Error

Error
  ▲
  │     Underfitting         Optimal         Overfitting
  │         Zone              Zone              Zone
  │          ╱───╲                            
  │         ╱     ╲                          ╱
  │        ╱       ╲                        ╱
  │       ╱         ╲_____________________ ╱  ← Test Error
  │      ╱                                ╱
  │     ╱                                ╱
  │    ╱                    ____________╱      ← Training Error
  │   ╱____________________╱
  │
  └────────────────────────────────────────────► Model Complexity
     Simple                              Complex
```

---

### Numerical Example: Polynomial Regression

**Dataset:** y = x² + noise

| x  | y (actual) |
|----|------------|
| 1  | 1.1        |
| 2  | 4.2        |
| 3  | 8.9        |
| 4  | 16.1       |
| 5  | 25.3       |

**Model Comparison:**

1. **Underfitting (Degree 1 - Linear):**
```
ŷ = 4.8x - 3.7

Training MSE: 25.6
Test MSE: 27.3
→ Poor fit, misses quadratic pattern
```

2. **Good Fit (Degree 2 - Quadratic):**
```
ŷ = 0.98x² + 0.1x + 0.05

Training MSE: 0.8
Test MSE: 1.2
→ Captures true pattern
```

3. **Overfitting (Degree 10 - Very High):**
```
ŷ = 0.001x¹⁰ - 0.05x⁹ + ... + 1.02x² + ...

Training MSE: 0.01
Test MSE: 145.7
→ Fits training perfectly, fails on new data
```

---

## Techniques to Mitigate Problems

### Techniques to Address Underfitting

#### 1. **Increase Model Complexity**

Add more parameters or use more sophisticated models.

**Implementation:**
```
Before: Linear Regression (y = β₀ + β₁x)
After: Polynomial Regression (y = β₀ + β₁x + β₂x² + β₃x³)

Example:
- Decision Tree: Increase max_depth
- Neural Network: Add more layers/neurons
- SVM: Use non-linear kernels (RBF instead of linear)
```

**Numerical Example:**
```
Linear model (underfitting): Training MSE = 45, Test MSE = 47
Polynomial model: Training MSE = 12, Test MSE = 15 ✓
```

---

#### 2. **Add More Features**

Include additional relevant features or create new ones.

**Feature Engineering:**
```
Original features: [size, bedrooms]
New features: [size, bedrooms, size², size×bedrooms, age, location]

Example:
price = β₀ + β₁(size) + β₂(bedrooms)  → Underfit
price = β₀ + β₁(size) + β₂(bedrooms) + β₃(size²) 
      + β₄(location) + β₅(age) → Better fit
```

---

#### 3. **Reduce Regularization**

Decrease the regularization parameter (λ).

**Formula:**
```
Before: J(β) = MSE + 10 Σβ²  (High regularization)
After:  J(β) = MSE + 0.1 Σβ² (Lower regularization)

Effect: Allows model to fit data more closely
```

**Numerical Example:**
```
λ = 10:  Training error = 30, Test error = 32 (Underfit)
λ = 1:   Training error = 15, Test error = 17 (Better)
λ = 0.1: Training error = 8, Test error = 10 (Good fit)
```

---

#### 4. **Train Longer**

Increase the number of training epochs or iterations.

**Implementation:**
```
Before: 10 epochs → Training accuracy 65%
After:  100 epochs → Training accuracy 85%

Note: Only if model hasn't converged yet
```

---

#### 5. **Remove Noise from Data**

Clean data to help model identify true patterns.

**Techniques:**
- Remove outliers
- Handle missing values properly
- Fix data entry errors
- Filter irrelevant samples

---

#### 6. **Use Better Algorithms**

Switch to more powerful algorithms.

**Progression:**
```
Linear Regression → Polynomial Regression
                 → Decision Trees
                 → Random Forests
                 → Gradient Boosting
                 → Neural Networks
```

---

### Techniques to Address Overfitting

#### 1. **Increase Training Data**

More data helps the model learn general patterns instead of memorizing.

**Effect:**
```
100 samples: Training = 95%, Test = 60% (Overfitting)
1000 samples: Training = 88%, Test = 85% (Good fit)
10000 samples: Training = 87%, Test = 86% (Better)

Rule of thumb: Need at least 10× more samples than parameters
```

**Numerical Example:**
```
Neural Network with 1000 parameters:
- 500 samples: Overfits (Test accuracy 65%)
- 5000 samples: Better (Test accuracy 82%)
- 50000 samples: Good (Test accuracy 88%)
```

---

#### 2. **Data Augmentation**

Artificially increase training data through transformations.

**For Images:**
```
Original image → Rotated versions
               → Flipped versions
               → Cropped versions
               → Brightness adjusted
               → With noise added

Example:
1000 images × 10 augmentations = 10,000 training samples
```

**For Text:**
```
- Synonym replacement
- Back translation
- Random insertion/deletion
```

---

#### 3. **Regularization (L1, L2, Elastic Net)**

Add penalty terms to prevent large weights.

**L2 Regularization (Ridge):**
```
J(β) = MSE + λ Σβⱼ²

Effect: Shrinks all weights toward zero

Example:
Without: β = [10, -8, 12, -15]
With L2 (λ=0.1): β = [3.2, -2.1, 3.8, -4.5]
```

**L1 Regularization (Lasso):**
```
J(β) = MSE + λ Σ|βⱼ|

Effect: Forces some weights to exactly zero (feature selection)

Example:
Without: β = [0.5, 0.3, 0.1, 0.05]
With L1 (λ=0.1): β = [0.4, 0.2, 0, 0]
```

**Numerical Comparison:**
```
No regularization:  Training = 99%, Test = 65% (Overfit)
L2 (λ=0.01):       Training = 92%, Test = 78%
L2 (λ=0.1):        Training = 88%, Test = 85% ✓
L2 (λ=1.0):        Training = 75%, Test = 73% (Starting to underfit)
```

---

#### 4. **Dropout (Neural Networks)**

Randomly deactivate neurons during training.

**Formula:**
```
During training:
  For each neuron with probability p:
    Output = 0
  
  For remaining neurons:
    Output = activation/(1-p)

During testing:
  All neurons active (no dropout)

Typical values: p = 0.2 to 0.5
```

**Implementation:**
```
Layer 1: 100 neurons, dropout = 0.3
→ Each iteration: ~30 neurons randomly turned off
→ Forces network to learn robust features

Effect:
Without dropout: Train=99%, Test=68%
With dropout(0.3): Train=89%, Test=86% ✓
```

**Numerical Example:**
```
Hidden layer with 5 neurons: [0.8, 0.6, 0.9, 0.7, 0.5]
Dropout mask (p=0.4): [1, 0, 1, 1, 0]
After dropout: [0.8/0.6, 0, 0.9/0.6, 0.7/0.6, 0]
            = [1.33, 0, 1.5, 1.17, 0]
```

---

#### 5. **Early Stopping**

Stop training when validation error starts increasing.

**Algorithm:**
```
1. Monitor validation error during training
2. Save model when validation error is lowest
3. Stop if validation error increases for N epochs
4. Return best saved model

Typical patience: 5-10 epochs
```

**Numerical Example:**
```
Epoch | Train Loss | Val Loss | Action
------|------------|----------|--------
1     | 2.50       | 2.60     | Continue
5     | 1.20       | 1.35     | Continue
10    | 0.80       | 0.95     | Save (best so far)
15    | 0.45       | 0.88     | Continue
20    | 0.25       | 0.92     | Continue (patience 1)
25    | 0.15       | 1.05     | Continue (patience 2)
30    | 0.08       | 1.15     | Stop! Return model from epoch 10
```

---

#### 6. **Cross-Validation**

Use multiple train-test splits to ensure robust performance.

**K-Fold Cross-Validation:**
```
Split data into K folds
For i = 1 to K:
    Train on K-1 folds
    Validate on fold i
    Record performance
Return average performance

Typical K: 5 or 10
```

**Numerical Example:**
```
5-Fold CV Results:
Fold 1: 84%
Fold 2: 88%
Fold 3: 82%
Fold 4: 87%
Fold 5: 86%

Average: 85.4%
Std Dev: 2.3%

High std dev might indicate overfitting to specific data subsets
```

---

#### 7. **Ensemble Methods**

Combine multiple models to reduce variance.

**Bagging (Bootstrap Aggregating):**
```
1. Create N bootstrap samples
2. Train model on each
3. Average predictions

Example - Random Forest:
Tree 1 prediction: 0.8
Tree 2 prediction: 0.6
Tree 3 prediction: 0.7
...
Tree 100 prediction: 0.75

Final prediction: average = 0.72
```

**Boosting:**
```
1. Train weak learner
2. Focus on misclassified samples
3. Add to ensemble
4. Repeat

Effect:
Single tree: Train=95%, Test=72%
Random Forest (100 trees): Train=88%, Test=86% ✓
```

**Numerical Example:**
```
Dataset with 100 samples:
Model A: 85 correct, 15 wrong
Model B: 82 correct, 18 wrong
Model C: 87 correct, 13 wrong

Ensemble (voting):
- Sample where A, B, C all agree: High confidence
- Sample where only 2 agree: Medium confidence
- Final accuracy: 89% (better than any individual)
```

---

#### 8. **Reduce Model Complexity**

Simplify the model architecture.

**Techniques:**
```
Neural Networks:
- Reduce number of layers: 10 → 5
- Reduce neurons per layer: 512 → 128
- Use simpler architecture

Decision Trees:
- Reduce max_depth: 20 → 5
- Increase min_samples_split: 2 → 20
- Limit number of features

Polynomial Regression:
- Lower degree: 10 → 3
```

**Numerical Example:**
```
Neural Network:
Architecture 1: [784, 512, 256, 128, 10] → Overfit
Architecture 2: [784, 128, 10] → Better fit

Results:
Architecture 1: Train=99%, Test=68%
Architecture 2: Train=91%, Test=88% ✓
```

---

#### 9. **Feature Selection**

Remove irrelevant or redundant features.

**Methods:**
```
1. Correlation analysis
2. Feature importance from trees
3. Recursive feature elimination
4. L1 regularization (automatic)
```

**Numerical Example:**
```
Original: 100 features
After selection: 25 most important features

Results:
100 features: Train=97%, Test=71% (Overfit)
25 features: Train=89%, Test=87% ✓

Bonus: Faster training, easier interpretation
```

---

#### 10. **Batch Normalization**

Normalize inputs to each layer in neural networks.

**Formula:**
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
y = γx̂ + β

Where:
- μ_batch = batch mean
- σ²_batch = batch variance
- γ, β = learnable parameters
- ε = small constant for stability
```

**Effect:**
- Reduces internal covariate shift
- Acts as regularization
- Allows higher learning rates
- Reduces dependence on initialization

**Numerical Example:**
```
Without batch norm: Train=99%, Test=65%
With batch norm: Train=93%, Test=89% ✓
```

---

### Decision Framework

```
┌─────────────────────────────────────────┐
│ Is your model performing poorly?        │
└────────────┬────────────────────────────┘
             │
             ▼
    ┌────────────────┐
    │ Check Metrics  │
    └────────┬───────┘
             │
             ▼
┌────────────────────────────────────────┐
│ Training Error HIGH, Test Error HIGH?  │ → YES → UNDERFITTING
│ (e.g., Train: 65%, Test: 63%)          │
└────────┬───────────────────────────────┘
         │ NO
         ▼
┌────────────────────────────────────────┐
│ Training Error LOW, Test Error HIGH?   │ → YES → OVERFITTING
│ (e.g., Train: 99%, Test: 68%)          │
└────────┬───────────────────────────────┘
         │ NO
         ▼
┌────────────────────────────────────────┐
│ Both errors LOW and similar?           │ → YES → GOOD FIT!
│ (e.g., Train: 88%, Test: 86%)          │
└────────────────────────────────────────┘
```

---

### Summary Table: When to Use Each Technique

| Problem | Technique | When to Use | Priority |
|---------|-----------|-------------|----------|
| **Underfitting** | Increase complexity | Model too simple | High |
| | Add features | Missing important patterns | High |
| | Reduce regularization | λ too high | Medium |
| | Train longer | Not converged | Medium |
| **Overfitting** | More training data | Always beneficial | Highest |
| | Regularization (L1/L2) | Many parameters | High |
| | Dropout | Deep neural networks | High |
| | Early stopping | Long training | High |
| | Cross-validation | Model selection | High |
| | Reduce complexity | Model too complex | Medium |
| | Feature selection | Too many features | Medium |
| | Ensemble methods | Reduce variance | Medium |
| | Data augmentation | Limited data | Medium |
| | Batch normalization | Deep networks | Low |

---

## Practical Guidelines

### Model Selection Checklist

```
✓ Start simple (baseline model)
✓ Evaluate with cross-validation
✓ Plot learning curves
✓ Check for under/overfitting
✓ Apply appropriate techniques
✓ Monitor train/test metrics
✓ Iterate and improve
```

### Learning Curves

**Underfitting Pattern:**
```
Error
  ▲
  │  ┌──────────────────  ← Training Error (High)
  │  │
  │  │
  │  └──────────────────  ← Validation Error (High)
  │
  └────────────────────────► Training Set Size
  
Both errors high and converged
```

**Overfitting Pattern:**
```
Error
  ▲
  │        ┌──────────────  ← Validation Error (High)
  │       ╱
  │      ╱
  │     ╱
  │  ──┘                   ← Training Error (Low)
  │
  └────────────────────────► Training Set Size
  
Large gap between errors
```

**Good Fit Pattern:**
```
Error
  ▲
  │  ┌──────────
  │  │╲        ╲
  │  │ ╲        ╲──────── ← Validation Error (Low)
  │  │  ╲        
  │  │   ╲───────────────  ← Training Error (Low)
  │
  └────────────────────────► Training Set Size
  
Small gap, both low
```

---

## Conclusion

Understanding numerical concepts and the bias-variance tradeoff is fundamental to machine learning success. The key is to:

1. **Diagnose the problem** correctly (underfitting vs overfitting)
2. **Apply appropriate techniques** based on the diagnosis
3. **Monitor performance** using proper metrics
4. **Iterate** until achieving the best balance

Remember: **Perfect training accuracy is not the goal** - generalization to unseen data is what matters.

---

## Quick Reference

### Underfitting Solutions
- ✅ Increase model complexity
- ✅ Add more features
- ✅ Reduce regularization
- ✅ Train longer
- ✅ Use better algorithms

### Overfitting Solutions
- ✅ Get more training data
- ✅ Use regularization (L1/L2)
- ✅ Apply dropout
- ✅ Early stopping
- ✅ Cross-validation
- ✅ Reduce model complexity
- ✅ Feature selection
- ✅ Data augmentation
- ✅ Ensemble methods

---

**Last Updated:** April 2026
