# ML-Model-from-Scratch

This repository contains a collection of machine learning models implemented from scratch using Python. The models are created without relying on external libraries, showcasing the underlying principles and algorithms involved in each model.

The implementation of these models is inspired by [Assembly AI: ML Models from Scratch](https://youtube.com/playlist?list=PLcWfeUsAys2k_xub3mHks85sBHZvg24Jd), a YouTube playlist that provides valuable insights into machine learning algorithms and their implementation.

This project is also inspired by Patrick Loeber, one of the demonstrators of Assembly AI. You can find more of his content on his [YouTube Channel](https://www.youtube.com/@patloeber).


## Instructions

To use the models in this repository, follow these steps:

1. **Install Dependencies:** Most of the models require the `numpy` and `matplotlib` libraries. Make sure you have both libraries installed before using the models. You can install them using the package manager of your choice (e.g., pip, conda).

2. **Model Usage:** Each model is contained within a separate file, making it easy to access and understand. Simply import the desired model file into your Python environment to access the model's functionalities.

3. **Explore and Experiment:** Feel free to explore the code and experiment with the models. Modify the code, tweak hyperparameters, and observe the results to gain a deeper understanding of how each model works.

4. **Further Customization:** As the models are implemented from scratch, you have the flexibility to modify and extend them according to your specific requirements. Dive into the code and adapt it to your use cases, or use it as a foundation to develop more advanced models.

By using these models, you can gain a solid understanding of the underlying algorithms and concepts in machine learning. Enjoy exploring and building upon these implementations to enhance your knowledge and skills in the field of machine learning!

## List of Machine Learning Models

### Supervised Learning

- **Decision Tree:** A decision tree is a flowchart-like model that uses a tree structure to make decisions by splitting data based on feature values. It can be used for both classification and regression tasks.

- **K-Nearest Neighbors (KNN):** KNN is a non-parametric algorithm that classifies data based on its proximity to labeled examples. It assigns a class label based on the majority vote of its k nearest neighbors.

- **Linear Regression:** Linear regression is a parametric algorithm used for predicting continuous numeric values. It establishes a linear relationship between the input variables and the target variable.

- **Logistic Regression:** Logistic regression is a classification algorithm used for binary or multi-class classification. It models the probability of a binary outcome based on the input variables using a logistic function.

- **Naive Bayes:** Naive Bayes is a probabilistic classifier that applies Bayes' theorem with the assumption of independence between features. It is commonly used for text classification and spam filtering.

- **Random Forest:** Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It uses a technique called bagging, where each tree is trained on a random subset of the data and features.

   Random Forest is particularly effective in handling high-dimensional datasets and reducing overfitting. It leverages the diversity of individual decision trees to make predictions by aggregating their results.

- **AdaBoost:** AdaBoost, short for Adaptive Boosting, is another ensemble learning method that combines multiple weak classifiers to create a strong classifier. It iteratively adjusts the weights of misclassified samples to focus on the difficult examples.

   AdaBoost assigns higher weights to misclassified samples in each iteration, which allows subsequent weak classifiers to prioritize them. By combining the predictions of multiple weak classifiers, AdaBoost improves overall accuracy.

### Unsupervised Learning

- **Principal Component Analysis (PCA):** PCA is a dimensionality reduction technique used to identify patterns and reduce the number of features in a dataset while preserving the most important information.

- **K-Means:** K-Means is a clustering algorithm that partitions data into k clusters based on their similarity. It aims to minimize the within-cluster sum of squares.

- **Support Vector Machines (SVM):** SVM is a classification algorithm that finds an optimal hyperplane in a high-dimensional feature space to separate different classes. It can also be used for regression and anomaly detection tasks.

- **Perceptron:** Perceptron is a binary classification algorithm used to classify linearly separable data. It is a single-layer neural network that learns weights for input features to make predictions.



