import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge datasets to create a comprehensive customer profile
customer_data = transactions.merge(customers, on='CustomerID')

# Feature Engineering
customer_features = customer_data.groupby('CustomerID').agg(
    TotalSpending=('TotalValue', 'sum'),
    AverageTransactionValue=('TotalValue', 'mean'),
    PurchaseFrequency=('TransactionID', 'count')
).reset_index()

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(customer_features[['TotalSpending', 'AverageTransactionValue', 'PurchaseFrequency']])

# Calculate cosine similarity
similarity_matrix = cosine_similarity(features_scaled)

# Create a DataFrame for similarity scores
similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

# Function to get top 3 lookalikes for a given customer
def get_lookalikes(customer_id, top_n=3):
    similar_scores = similarity_df[customer_id].sort_values(ascending=False)
    lookalikes = similar_scores[similar_scores.index != customer_id].head(top_n)
    return lookalikes.index.tolist(), lookalikes.values.tolist()

# Generate lookalikes for the first 20 customers
lookalike_results = {}
for customer_id in customer_features['CustomerID'].head(20):
    lookalikes, scores = get_lookalikes(customer_id)
    lookalike_results[customer_id] = list(zip(lookalikes, scores))

# Flatten the results into a list of dictionaries
flattened_results = []
for cust_id, lookalikes in lookalike_results.items():
    for lookalike, score in lookalikes:
        flattened_results.append({
            'CustomerID': cust_id,
            'Lookalike_CustomerID': lookalike,
            'Similarity_Score': score
        })

# Convert results to DataFrame
lookalike_df = pd.DataFrame(flattened_results)

# Save the lookalike results to a CSV file
lookalike_df.to_csv('Lookalike.csv', index=False)

print("Lookalike model results saved to 'Lookalike.csv'.")