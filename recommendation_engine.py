from flask import Flask, request, jsonify
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js backend

# Global variables to store model data
df = None
interaction_matrix = None
matrix_factors = None
svd = None

def load_model():
    """Load and train the recommendation model"""
    global df, interaction_matrix, matrix_factors, svd
    
    try:
        logger.info("Loading dataset and training model...")
        df = pd.read_csv('user_book_interactions.csv')
        
        # Create user-book interaction matrix
        interaction_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        sparse_matrix = csr_matrix(interaction_matrix.values)
        
        # Apply SVD for collaborative filtering
        svd = TruncatedSVD(n_components=10, random_state=42)
        matrix_factors = svd.fit_transform(sparse_matrix)
        
        logger.info(f"Model loaded successfully! Users: {len(interaction_matrix.index)}, Books: {len(interaction_matrix.columns)}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/recommend', methods=['GET'])
def recommend():
    """Get personalized book recommendations for a user"""
    try:
        # Get parameters from request
        user_id_str = request.args.get('user_id')
        top_n = int(request.args.get('top_n', 5))
        
        logger.info(f"Recommendation request for user_id: {user_id_str}, top_n: {top_n}")
        
        # Check if model is loaded
        if interaction_matrix is None:
            if not load_model():
                return jsonify({
                    "success": False,
                    "message": "Recommendation model not available",
                    "recommendations": []
                }), 500
        
        # Handle MongoDB ObjectId conversion if needed
        try:
            # If numerical user ID
            user_id = int(user_id_str)
        except ValueError:
            # If MongoDB ObjectId format, check if we have a mapping
            # For now, return an error since we don't have a mapping implemented
            return jsonify({
                "success": False, 
                "message": "User not found in recommendation system",
                "recommendations": []
            }), 404
        
        # Check if user exists in model
        if user_id not in interaction_matrix.index:
            logger.warning(f"User {user_id} not found in recommendation system")
            return jsonify({
                "success": False,
                "message": "User not found in recommendation system",
                "recommendations": []
            }), 404
        
        # Get user's position in the matrix
        user_index = interaction_matrix.index.get_loc(user_id)
        user_factors = matrix_factors[user_index]
        
        # Calculate predicted ratings
        predicted_ratings = matrix_factors.dot(user_factors.T)
        
        # Get books the user has already interacted with
        user_interactions = df[df['user_id'] == user_id]['book_id'].values
        
        # Generate recommendations excluding books user has already interacted with
        recommended_books = pd.Series(predicted_ratings, index=interaction_matrix.columns)
        recommended_books = recommended_books[~recommended_books.index.isin(user_interactions)]
        recommended_books = recommended_books.sort_values(ascending=False).head(top_n)
        
        # Convert recommendations to list
        recommendations = recommended_books.index.tolist()
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({
            "success": False,
            "message": f"Error generating recommendations: {str(e)}",
            "recommendations": []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "healthy" if interaction_matrix is not None else "model not loaded"
    users = len(interaction_matrix.index) if interaction_matrix is not None else 0
    books = len(interaction_matrix.columns) if interaction_matrix is not None else 0
    
    return jsonify({
        "status": status, 
        "users": users,
        "books": books
    }), 200 if status == "healthy" else 503

@app.route('/refresh', methods=['POST'])
def refresh_model():
    """Endpoint to refresh the recommendation model"""
    success = load_model()
    return jsonify({
        "success": success,
        "message": "Model refreshed successfully" if success else "Failed to refresh model"
    }), 200 if success else 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    logger.info(f"Starting recommendation service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
