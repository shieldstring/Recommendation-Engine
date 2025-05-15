import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model variables
model_initialized = False
interaction_matrix = None
matrix_factors = None
svd = None

def initialize_model():
    """Initialize or re-initialize the recommendation model"""
    global model_initialized, interaction_matrix, matrix_factors, svd
    
    try:
        logger.info("Initializing recommendation model...")
        
        # Generate sample data (replace this with your actual data loading)
        num_users = 100
        num_books = 50
        np.random.seed(42)
        
        # Create sample user-book interactions
        user_ids = np.repeat(np.arange(1, num_users + 1), 10)
        book_ids = np.random.choice(np.arange(1, num_books + 1), num_users * 10)
        ratings = np.random.randint(1, 6, num_users * 10)
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'book_id': book_ids,
            'rating': ratings
        })
        
        # Create interaction matrix
        interaction_matrix = df.pivot_table(
            index='user_id',
            columns='book_id',
            values='rating',
            fill_value=0
        )
        
        # Train SVD model
        sparse_matrix = csr_matrix(interaction_matrix.values)
        svd = TruncatedSVD(n_components=min(10, sparse_matrix.shape[1] - 1), random_state=42)
        matrix_factors = svd.fit_transform(sparse_matrix)
        
        model_initialized = True
        logger.info(f"Model initialized with {len(interaction_matrix)} users and {len(interaction_matrix.columns)} books")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        if not model_initialized:
            initialize_model()
            
        user_id = int(request.args.get('user_id', 1))
        top_n = int(request.args.get('top_n', 5))
        
        if user_id not in interaction_matrix.index:
            return jsonify({
                "success": False,
                "message": f"User {user_id} not found",
                "recommendations": []
            }), 404
            
        user_idx = interaction_matrix.index.get_loc(user_id)
        user_vector = matrix_factors[user_idx]
        scores = matrix_factors.dot(user_vector)
        
        # Get top N recommendations
        recommended_indices = (-scores).argsort()[:top_n]
        recommendations = interaction_matrix.columns[recommended_indices].tolist()
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e),
            "recommendations": []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model_initialized else "initializing",
        "users": len(interaction_matrix) if model_initialized else 0,
        "books": len(interaction_matrix.columns) if model_initialized else 0
    })

if __name__ == '__main__':
    initialize_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)