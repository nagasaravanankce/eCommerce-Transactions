# eCommerce-Transactions

This project involves analyzing an eCommerce transactions dataset to perform customer segmentation using clustering techniques. The goal is to group customers based on their purchasing behavior to enable targeted marketing strategies.

## Dataset
The project uses three datasets:
- **Customers.csv**: Contains customer information including ID, name, region, and signup date.
- **Products.csv**: Contains product information including ID, name, category, and price.
- **Transactions.csv**: Contains transaction data including transaction ID, customer ID, product ID, transaction date, quantity, and total value.

## Tasks Completed
1. **Exploratory Data Analysis (EDA)**:
   - Performed EDA to understand the dataset and derive business insights.
   - Visualized data using basic plots and identified key trends.

2. **Lookalike Model**:
   - Built a lookalike model to recommend similar customers based on their profiles and transaction history.
   - Used cosine similarity to find similar customers.

3. **Customer Segmentation**:
   - Implemented K-Means clustering to segment customers based on their purchasing behavior.
   - Evaluated clustering results using metrics such as the Davies-Bouldin Index and Silhouette Score.

