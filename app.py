from flask import Flask, request, jsonify
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Load dataset and train model
df = pd.read_csv('user_book_interactions.csv')
interaction_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
sparse_matrix = csr_matrix(interaction_matrix.values)
svd = TruncatedSVD(n_components=10, random_state=42)
matrix_factors = svd.fit_transform(sparse_matrix)

# Recommendation endpoint
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    top_n = int(request.args.get('top_n', 5))

    user_index = interaction_matrix.index.get_loc(user_id)
    user_factors = matrix_factors[user_index]
    predicted_ratings = matrix_factors.dot(user_factors.T)
    recommended_books = pd.Series(predicted_ratings, index=interaction_matrix.columns)
    recommended_books = recommended_books.sort_values(ascending=False).head(top_n)
    recommendations = recommended_books.index.tolist()

    return jsonify({"user_id": user_id, "recommendations": recommendations})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)