# Exploratory Data Analysis of the CelebA Dataset

## Project Details

- **Name**: Artham Bhardwaj
- **Company**: CodTech IT Solutions 
- **ID**: CT12DS1704
- **Domain**: DATA SCIENCE
- **Duration**: July 10 , 2024 - Sep 10 , 2024
- **Mentor**: Mr. Muzammil Ahmed

## Overview of the Project

### Project Title
**Exploratory Data Analysis of the CelebA Dataset**

### Objective
The primary objective of this project is to perform an in-depth Exploratory Data Analysis (EDA) on the CelebA dataset, which contains facial images of celebrities along with 40 binary attributes. The goal is to uncover insights related to attribute distributions, correlations, and potential biases in the data, which will inform subsequent machine learning or data processing tasks.

### Key Activities
1. **Data Loading**: Import the CelebA attributes dataset into a pandas DataFrame.
2. **Data Exploration**: Examine the dataset's structure, check for missing values, and compute summary statistics.
3. **Attribute Distribution Analysis**: Visualize the distribution of key attributes such as gender, smiling, and wearing a hat.
4. **Correlation Analysis**: Generate a correlation matrix to identify relationships between different attributes.
5. **Outlier Detection**: Detect and visualize outliers in attributes using techniques like the IQR method.
6. **Visualization**: Create histograms, scatter plots, and heatmaps to effectively communicate the findings.

### Technology Used
- **Programming Language**: Python 3.x
- **Libraries**: 
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - opencv-python (optional, for image processing)

### Dataset Used
- **Dataset**: [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **Description**: The dataset includes over 200,000 images of celebrities, each labeled with 40 binary attributes such as gender, age, and facial features.

### Insights
- The dataset is relatively balanced in terms of gender distribution, with a slight male bias.
- Attributes like 'Young' and 'Smiling' are predominant, indicating a potential skew in the dataset.
- Some attributes, such as 'Heavy Makeup' and 'Wearing Lipstick', show strong correlations, while most other attributes are relatively independent.
- No significant outliers were found, indicating consistency in the data.

### Result and Conclusion
The EDA provided a clear understanding of the CelebA dataset's structure, revealing important insights into attribute distributions and relationships. The analysis confirmed the presence of potential biases, such as the prevalence of 'Young' and 'Smiling' labels, which should be considered in any future modeling efforts. The dataset's consistency, as evidenced by the lack of significant outliers, ensures reliability for further analysis.

