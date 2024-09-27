import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample dataset
df = sns.load_dataset('iris')

# Set the style for Seaborn
sns.set(style="whitegrid")

# 1. Line Plot using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(df['sepal_length'], label='Sepal Length')
plt.plot(df['sepal_width'], label='Sepal Width')
plt.title('Line Plot of Sepal Dimensions')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# 2. Scatter Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()

# 3. Histogram using Matplotlib
plt.figure(figsize=(10, 6))
plt.hist(df['petal_length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()

# 4. Box Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_length', data=df)
plt.title('Box Plot of Petal Length by Species')
plt.show()

# 5. Pair Plot using Seaborn
sns.pairplot(df, hue='species')
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# 6. Heatmap using Seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# 7. Bar Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal_length', data=df, ci=None)
plt.title('Bar Plot of Sepal Length by Species')
plt.show()


# Explanation:
# Line Plot: Shows trends over time or indices.
# Scatter Plot: Displays the relationship between two numerical variables.
# Histogram: Shows the distribution of a single numerical variable.
# Box Plot: Displays the distribution of a numerical variable across different categories.
# Pair Plot: Shows pairwise relationships in a dataset.
# Heatmap: Displays the correlation matrix of numerical variables.
# Bar Plot: Compares the average values of a numerical variable across different categories.