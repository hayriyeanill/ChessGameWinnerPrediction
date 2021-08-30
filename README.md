# Machine Learning and Pattern Recognition Class
# ChessGameWinnerPrediction

Chess is a very popular sport that is played frequently in the world and where tournaments are held. Chess
played with two people or on online platforms is a game that can be played both simply and pushing the limits
of the mind. Thanks to artificial intelligence, the ability of a computer to perform various activities similar to
living things, machines can play chess against humans. With the help of artificial intelligence, computers can
analyze human thinking methods to predict their opponent's moves or predict whether his opponent will win
the game based on players’ past games. As a branch of AI, Machine Learning is the use of algorithms to learn
from the data and make informed decision. In this project, a system that will evaluate the prediction results of
machine learning algorithms based on the players’ past games, take into account the algorithm that gives the
best performance and make a winner prediction based on the game is proposed.

# Dataset 
A dataset of more than 20,000 games collected from specific users on Lichess.org (online chess server) was
selected from Kaggle public resource [4]. The purpose of this dataset is to predict the winner based on
information such as the players' game history, game habits, how many moves they have won, and their scores.
Games with more than 20 moves were selected in the dataset. ID information has been removed from the dataset.
Dataset contains 18263 observations (rows) and 13 features (columns). The dataset contains 6 numerical 
features (created_at, last_move_at, turns, white_rating, black_rating, opening_ply) and 7 nominal features
(rated, victory_status, winner, increment_code, moves, opening_eco, opening_name) that were converted into
factors with numerical value designated for each level.

Available on: https://www.kaggle.com/datasnaek/chess

# Feature Selection Methods
1) Backward Elimination
2) Forward Selection

# Feature Extraction Methods
1 - Principal Components Analysis (PCA)
2 - Linear Discriminant Analysis (LDA)

# Classification Methods
1 - Logistic Regression
2 -  Decision Tree Classifier
3 - Random Forest Classifier
4 - K – Nearest Neighbor
5 -  Support Vector Machine
6 - Gaussian Naive Bayes
7 - Multi-class Classification with Neural Networks

# Measuring the Performance of Classification Models:
1 - Confussion Matrix
2 - Accuracy
3 - Error Rate / Misclassification Rate
