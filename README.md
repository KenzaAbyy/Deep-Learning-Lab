# Deep Learning Lab — Regression and Multiclass Classification

This repository contains my work for a Deep Learning lab divided into two parts. The first part focuses on stock price prediction using regression, while the second part deals with predictive maintenance using multiclass classification.

For both parts, I used **PyTorch** to build and train neural networks, and **Scikit-learn** to preprocess the data, evaluate the models, and perform hyperparameter tuning.

My goal was not only to obtain good results, but also to understand how different architectures, hyperparameters, and regularization techniques affect model performance.

## Part 1 — Stock Price Regression (NYSE)

### Dataset and objective

For this part, I used the NYSE stock prices dataset and focused on Apple (AAPL).

The objective was to predict the **next trading day's closing price** using five features from the current day: opening price, highest price, lowest price, closing price, and trading volume.

After preparing the data, I obtained 1,761 observations.

### What I did

I started with exploratory data analysis to understand the dataset. I visualized stock price distributions, checked correlations between features, identified outliers, and plotted the evolution of Apple's closing price.

I noticed that the price-related features were strongly correlated. I also observed a sudden price change in 2014 related to Apple's stock split. I decided to keep the original data for this experiment, although this is a limitation of the dataset.

Next, I prepared the data for regression. Since this is time-series data, I used a chronological split rather than randomly shuffling the observations. I also applied standardization before training.

I implemented a Multilayer Perceptron (MLP) using PyTorch, with fully connected layers and ReLU activation functions. The model was trained using MSE loss and the Adam optimizer.

I then used GridSearchCV with TimeSeriesSplit to compare different architectures, learning rates, optimizers, and numbers of epochs.

Finally, I experimented with Dropout, L2 regularization, and Early Stopping to see whether these techniques could improve the model's generalization.

### Final results

The final selected model was an MLP with L2 regularization.

| Metric | MLP + L2 | Naive Baseline |
|---|---:|---:|
| RMSE | 1.6671 | 1.5043 |
| MAE | 1.2390 | 1.0635 |
| R² | 0.9505 | 0.9597 |

The model achieved an R² of approximately 0.95 on the test set, which shows that it followed the general evolution of stock prices quite well.

However, the naive baseline, which simply predicts that tomorrow's closing price will be equal to today's, performed slightly better.

This was an interesting result for me because it showed that a more complex model does not necessarily give better predictions.

### What I learned

This part helped me understand how to prepare time-series data for a neural network, implement regression with PyTorch, and evaluate predictions using RMSE, MAE, and R².

I also learned why chronological splitting is important, how GridSearchCV can help select hyperparameters, and how regularization can affect model performance.

Most importantly, I understood the value of comparing a neural network with a simple baseline instead of relying only on its evaluation metrics.

---

## Part 2 — Predictive Maintenance Classification

### Dataset and objective

For the second part, I worked with the Machine Predictive Maintenance Classification dataset, containing 10,000 observations.

The objective was to classify machine conditions into six categories:

- No Failure
- Heat Dissipation Failure
- Power Failure
- Overstrain Failure
- Tool Wear Failure
- Random Failures

The main challenge was the strong imbalance between classes. Most observations corresponded to normal machine operation, while some failure types had very few examples.

### Data preprocessing and exploration

I first checked the dataset for missing values and duplicates. I also examined the feature types and the distribution of failure categories.

I removed identifier columns and the binary `Target` column, since the objective was to predict the multiclass `Failure Type` label.

I encoded the categorical `Type` feature, encoded the target labels, and standardized the numerical data.

I used a stratified train/test split to preserve the class distribution, keeping 20% of the observations for the final test.

To address class imbalance, I applied **SMOTENC** to increase the number of minority-class examples in the training data. I kept the validation and test observations separate from this augmentation process.

### Deep Neural Network

I implemented a fully connected DNN using PyTorch, with ReLU activation functions and CrossEntropyLoss for multiclass classification.

During training, I monitored the loss and accuracy curves to understand how the model was learning.

The initial model achieved approximately 96.69% accuracy and a macro F1-score of 0.5739 on the validation set.

However, I noticed that some minority classes were still difficult to identify.

### Hyperparameter tuning

I used GridSearchCV with three-fold stratified cross-validation to test 24 hyperparameter combinations.

I compared different hidden-layer architectures, learning rates, optimizers (Adam and SGD), and training epochs.

The best configuration found was:

| Hyperparameter | Selected value |
|---|---|
| Hidden layers | (128, 64) |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Epochs | 50 |

The best average macro F1-score during cross-validation was **0.5683**.

When I evaluated the optimized model on the validation set, its macro F1-score improved to **0.5962**.

### Regularization experiments

I then compared four configurations: a DNN without regularization, a DNN with Dropout, a DNN with L2 regularization, and a DNN combining Dropout, L2, and Early Stopping.

| Model | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| DNN without regularization | 0.9769 | 0.6150 |
| DNN + Dropout | 0.9675 | 0.6161 |
| DNN + L2 | 0.9719 | 0.6112 |
| DNN + Dropout + L2 + Early Stopping | 0.9594 | 0.5434 |

Dropout achieved the highest macro F1-score, although the difference compared with the unregularized model was small.

I selected the DNN with Dropout (0.2) for the final evaluation.

### Final test results

After retraining the selected configuration, I evaluated it on the 2,000 test observations.

| Metric | Majority Baseline | DNN + Dropout |
|---|---:|---:|
| Accuracy | 0.9650 | 0.9620 |
| Precision Macro | 0.1608 | 0.5291 |
| Recall Macro | 0.1667 | 0.5702 |
| F1-score Macro | 0.1637 | 0.5470 |

At first, the accuracy of 96.20% seemed very good. However, the majority baseline achieved 96.50% accuracy simply by predicting `No Failure` for every observation.

This made me realize how misleading accuracy can be when classes are highly imbalanced.

The macro F1-score gives a more useful picture. My DNN performed much better than the baseline on this metric because it learned to identify several failure types.

The final confusion matrix showed that the model correctly detected 20 out of 22 Heat Dissipation failures, 12 out of 16 Overstrain failures, and 15 out of 19 Power failures.

However, it failed to detect the Random Failures and Tool Wear Failure examples in the test set.

### What I learned

This part helped me understand how to build a multiclass classification model with PyTorch and how to handle imbalanced datasets.

I learned how to apply SMOTENC, perform hyperparameter optimization, and compare regularization techniques.

The most important lesson was that **high accuracy does not always mean good classification performance**, especially when the dataset contains rare classes.

I also learned to pay more attention to precision, recall, macro F1-score, and confusion matrices when evaluating a classification model.

---

## Tools and Libraries

- Python
- Google Colab
- PyTorch
- Scikit-learn
- Imbalanced-learn
- Pandas and NumPy
- Matplotlib and Seaborn

## Datasets

**Part 1:** NYSE Stock Prices dataset, using Apple (AAPL) observations.

**Part 2:** [Machine Predictive Maintenance Classification — Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictivemaintenance-classification)

## Conclusion

Through these two exercises, I became more comfortable with the complete deep learning workflow, from exploring and preparing a dataset to building, training, optimizing, and evaluating neural networks.

I also realized that obtaining a high score is not the only objective. Understanding the errors, choosing the right metrics, and comparing results with simple baselines are just as important.

Both experiments showed me that neural networks can learn useful patterns, but their performance depends strongly on the data, the evaluation method, and the choices made during training.
