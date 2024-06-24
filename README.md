# __Heart Attack Prediction: Exploring Machine Learning Classifiers__

This project applies various classification algorithms to predict heart attacks based on a given dataset. The project is implemented in Jupyter notebooks.

## Classification Algorithms Used

The following machine learning algorithms are employed in this project:

- **Gaussian Naive Bayes**: This is a probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.

- **K-Nearest Neighbors (KNN)**: KNN is a non-parametric method used for classification and regression. The output is a class membership.

- **Radius Nearest Neighbors**: This is a variant of KNN where the neighbors are defined as all samples lying within a fixed radius r of the query point.

- **Decision Tree**: Decision tree builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed.

- **Support Vector Machine (SVM)**: SVM is a supervised learning model with associated learning algorithms that analyze data used for classification and regression analysis.

- **Logistic Regression**: Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.



## Evaluation and Results

In this project, all classification methods are rigorously evaluated using cross-validation. Given the critical nature of medical predictions, we have chosen to use **Recall** and **F1 scores** as our primary evaluation metrics, as they provide a more balanced view of our model's performance than accuracy alone.

The results of each algorithm are visualized through plots, allowing for an intuitive comparison of their performance. These visualizations provide valuable insights into the strengths and weaknesses of each algorithm in predicting heart attacks.


## Usage

To use this notebook, you can install Python, along with the necessary libraries (numpy, pandas, matplotlib, scikit-learn, seaborn). You can then run the notebook cell-by-cell to follow along with the analysis. Alternatively the file can be run on https://colab.research.google.com/



## License

MIT License

![image](https://github.com/YosefCh/Heart-Attack-Classifiction-Methods/assets/155560788/eacedec6-52b8-4736-99cf-f28a23edd789)

