
# Task
Analyze the Breast Cancer Wisconsin (Diagnostic) Dataset using PCA to reduce dimensionality to 2 components, visualize the results to identify clusters, and determine the number of components needed to explain most of the variance.

## Load the dataset

### Subtask:
Load the Breast Cancer Wisconsin (Diagnostic) Dataset from scikit-learn.

**Reasoning**:
Import the necessary function and load the dataset.
"""

from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()

"""## Scale the data

### Subtask:
Scale the features of the dataset using `StandardScaler`.

**Reasoning**:
Scale the feature data using StandardScaler as per the instructions.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(breast_cancer_data.data)

"""## Apply pca

### Subtask:
Apply PCA to reduce the dimensionality to 2 components.

**Reasoning**:
Apply PCA to the scaled data to reduce dimensionality to 2 components.
"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

"""## Visualize the results

### Subtask:
Create a scatter plot of the two principal components, coloring the points by the tumor type (benign or malignant).

**Reasoning**:
The instructions require creating a scatter plot of the two principal components, colored by the tumor type, with appropriate labels and a legend. This can be achieved with a single code block using matplotlib.
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=breast_cancer_data.target, cmap='viridis')
plt.title("PCA of Breast Cancer Dataset (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
legend_labels = breast_cancer_data.target_names
handles, _ = scatter.legend_elements()
plt.legend(handles, legend_labels, title="Tumor Type")
plt.grid(True)
plt.show()

"""## Investigate the number of components

### Subtask:
Determine how many components are needed to explain a significant portion of the variance in the data.

**Reasoning**:
Re-apply PCA without specifying the number of components to calculate explained variance, calculate cumulative explained variance, plot the cumulative explained variance, and use the plot to determine the number of components needed to explain a significant portion of the variance.
"""

pca_full = PCA()
pca_full.fit(scaled_data)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance Explained')
plt.legend()
plt.show()

# Determine the number of components for 95% variance
n_components_95 = (cumulative_explained_variance >= 0.95).argmax() + 1
print(f"Number of components needed to explain at least 95% variance: {n_components_95}")

"""## Summary:

### Data Analysis Key Findings

*   A scatter plot of the first two principal components shows a clear separation between the points representing benign and malignant tumors, suggesting that these two components capture significant discriminatory information.
*   Approximately 10 components are needed to explain at least 95% of the total variance in the dataset.

### Insights or Next Steps

*   The strong separation observed in the 2-component PCA plot indicates that even with significant dimensionality reduction, PCA retains features that distinguish between benign and malignant tumors. This suggests that PCA could be a valuable preprocessing step for building classification models on this dataset.
*   While 2 components provide a visually interpretable separation, using around 10 components could be more effective for machine learning models aiming for higher accuracy, as this retains 95% of the data's variance. The choice of the number of components for a model would depend on the trade-off between model complexity and desired performance.

"""
