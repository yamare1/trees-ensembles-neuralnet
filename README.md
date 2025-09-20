# Model Comparison Framework: Trees, Ensembles, and Neural Networks

Comprehensive analysis comparing decision trees, ensemble methods, and neural networks on structured tabular data. Demonstrates systematic model evaluation and selection techniques applicable to quantitative modeling and risk assessment.

## Summary

Developed a framework for comparing ML model families, achieving optimal performance with Gradient Boosting (86% accuracy) while documenting computational trade-offs and overfitting characteristics crucial for production deployment.

## Key Findings

### Performance Results
| Model | Accuracy | Training Time | Overfitting Risk |
|-------|----------|---------------|------------------|
| Decision Tree | 83.0% | 2.0s | High |
| Random Forest | 85.0% | 90.0s | Low |
| AdaBoost | 86.0% | 30.0s | Low |
| Gradient Boosting | 86.0% | 40.0s | Low |
| Neural Network | 85.1% | 162.2s | Medium |

### Technical Insights
- Tree ensembles outperformed neural networks on structured data
- Identified optimal depth=5 for single trees before overfitting
- Neural networks required 81× more tuning time than decision trees
- Boosting methods achieved best accuracy-efficiency trade-off

## Project Components

### 1. Overfitting Analysis
Systematic investigation of model complexity vs generalization:
- Decision trees: Severe overfitting at unlimited depth
- Random forests: Inherent resistance through bagging
- Neural networks: Architecture-dependent overfitting patterns

### 2. Hyperparameter Optimization
- Implemented RandomizedSearchCV with stratified k-fold validation
- Explored 270+ hyperparameter combinations
- Documented optimal configurations for each model type

### 3. Computational Efficiency
Benchmarked training and inference times across model families:
- Decision trees: Fastest training, instant inference
- Ensemble methods: Moderate training, parallelizable
- Neural networks: Slowest training, batch-optimized inference

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/trees-ensembles-neuralnet.git
cd model-comparison-framework

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
