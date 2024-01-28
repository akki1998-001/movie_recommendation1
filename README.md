# movie_recommendation1
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use Singular Value Decomposition (SVD) for collaborative filtering
algo = SVD()

# Train the model
algo.fit(trainset)

# Make predictions on the testing set
predictions = algo.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Function to get top N recommendations for a user
def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Get top recommendations for a specific user
user_id = '1'
user_predictions = [pred for pred in predictions if pred.uid == user_id]
top_n_recommendations = get_top_n_recommendations(user_predictions)
print("Top recommendations for user", user_id)
for movie_id, rating in top_n_recommendations[user_id]:
    print("Movie ID:", movie_id, "| Rating:", rating)
