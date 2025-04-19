import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Load dataset
df = pd.read_csv('user_book_interactions.csv')

# Create a user-book interaction matrix
interaction_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
sparse_matrix = csr_matrix(interaction_matrix.values)

# Apply SVD for collaborative filtering
svd = TruncatedSVD(n_components=10, random_state=42)
matrix_factors = svd.fit_transform(sparse_matrix)

# Function to generate recommendations for a user
def get_recommendations(user_id, top_n=5):
    user_index = interaction_matrix.index.get_loc(user_id)
    user_factors = matrix_factors[user_index]
    predicted_ratings = matrix_factors.dot(user_factors.T)
    recommended_books = pd.Series(predicted_ratings, index=interaction_matrix.columns)
    recommended_books = recommended_books.sort_values(ascending=False).head(top_n)
    return recommended_books.index.tolist()

# Example: Get recommendations for user_id = 1
print(get_recommendations(1))