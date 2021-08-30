import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("games.csv")

# DATA PRE PROCESSING
data['opening_eco'] = pd.factorize(data['opening_eco'])[0]
data['increment_code'] = pd.factorize(data['increment_code'])[0]

# rated, victory_status, winner column to label encoder
le = LabelEncoder()
data['rated'] = le.fit_transform(data['rated'])
data['victory_status'] = le.fit_transform(data['victory_status'])
data['winner'] = le.fit_transform(data['winner'])
class_names = {index: label for index, label in enumerate(le.classes_)}

# count number of moves
data['moves'] = data['moves'].str.count(" ") + 1

# Get games with more than 20 moves
data = data[data["moves"] > 20]
data = data.reset_index()

data = data[['rated', 'created_at', 'last_move_at', 'turns', 'victory_status', 'increment_code',
             'white_rating', 'black_rating', 'moves', 'opening_eco', 'opening_ply', 'winner']]

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Scale x features
scx = StandardScaler()
X_scale = scx.fit_transform(x)

# PCA
pca = PCA(n_components=11)
pca.fit(X_scale)
print("pca.explained_variance_ratio", pca.explained_variance_ratio_)
print("sum of pca.explained_variance_ratio", sum(pca.explained_variance_ratio_))
# Find optimum number of components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Explained variance")
plt.show()

print("Variance explained by the first principal components = ")
print(np.cumsum(pca.explained_variance_ratio_ * 100)[0])
print("Variance explained by the 2 principal components = ")
print(np.cumsum(pca.explained_variance_ratio_ * 100)[1])
print("Variance explained by the 3 principal components = ")
print(np.cumsum(pca.explained_variance_ratio_ * 100)[2])
print("Variance explained by the 7 principal components = ")
print(np.cumsum(pca.explained_variance_ratio_ * 100)[6])

# Apply PCA by setting n_components=2
pca_2 = PCA(n_components=2, random_state=2020)
pca_2.fit(X_scale)
X_pca_2 = pca_2.transform(X_scale)
plt.figure(figsize=(10,7))
sns.scatterplot(x=X_pca_2[:,0], y=X_pca_2[:,1], s=70, hue=data.winner, palette=['green', 'blue', 'pink'])
plt.xlabel("First principal")
plt.ylabel("Second principal")
plt.show()

# Apply optimum number of component
pca = PCA(n_components=7, random_state=2020)
pca.fit(X_scale)
pca_obj = pca.transform(X_scale)
pca_obj_df = pd.DataFrame(pca_obj)


def visualize_confussion_matrix(title,classifier, x_test, y_test, color):
    plot_confusion_matrix(classifier, x_test, y_test, display_labels=np.array(list(class_names.values())),
                          cmap=color, values_format='.1f')
    plt.title(title)
    plt.show()


def k_fold_cross_validation(x_samples, y_sample):
    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
    for train_index, test_index in kf.split(x_samples):
        x_train, x_test = x_samples[train_index], x_samples[test_index]
        y_train, y_test = y_sample[train_index], y_sample[test_index]
    return x_train, x_test, y_train, y_test


def train_test_split_method(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    return x_train, x_test, y_train, y_test


def logistic_regression(x_train, y_train, x_test, y_test):
    lrc = LogisticRegression(random_state=0, max_iter=1000)
    lrc.fit(x_train, y_train)
    lrc_pred = lrc.predict(x_test)
    cm_lrc = confusion_matrix(y_test, lrc_pred)
    acc_lrc = accuracy_score(y_test, lrc_pred)
    print("Logistic Regression", acc_lrc)
    title = "Logistic Regression Confussion Matrix"
    visualize_confussion_matrix(title, lrc, x_test, y_test, plt.cm.Blues)


def decision_tree(x_train, y_train, x_test, y_test):
    dtc = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)
    dtc.fit(x_train, y_train)
    dtc_pred = dtc.predict(x_test)
    cm_dtc = confusion_matrix(y_test, dtc_pred)
    acc_dtc = accuracy_score(y_test, dtc_pred)
    print("Decision Tree", acc_dtc)
    title = "Decision Tree Confussion Matrix"
    visualize_confussion_matrix(title, dtc, x_test, y_test, plt.cm.Greens)


def random_forest(x_train, y_train, x_test, y_test):
    rfc = RandomForestClassifier(n_estimators=50, max_depth=5, criterion='entropy', random_state=0)
    rfc.fit(x_train, y_train)
    rfc_pred = rfc.predict(x_test)
    cm_rfc = confusion_matrix(y_test, rfc_pred)
    acc_rfc = accuracy_score(y_test, rfc_pred)
    print("Random Forest", acc_rfc)
    title = "Random Forest Confussion Matrix"
    visualize_confussion_matrix(title, rfc, x_test, y_test, plt.cm.Reds)


def knn(x_train, y_train, x_test, y_test):
    # Find optimum k from error rate plot
    error_rate = []
    for i in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
        knn.fit(x_train, y_train)
        knn_pred = knn.predict(x_test)
        error_rate.append(np.mean(knn_pred != y_test))

    plt.plot(range(1, 11), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
             markersize=10)
    plt.title('Error Rate vs. k Value')
    plt.xlabel('k')
    plt.ylabel('Error Rate')
    plt.show()

    min_value = min(error_rate)
    min_index = error_rate.index(min_value) + 1  # index start from 0
    print("optimum k is ", min_index)
    # Apply optimum k
    knn = KNeighborsClassifier(n_neighbors=min_index, metric='euclidean')
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)
    cm_knn = confusion_matrix(y_test, knn_pred)
    acc_knn = accuracy_score(y_test, knn_pred)
    title = "K-NN Confussion Matrix"
    visualize_confussion_matrix(title, knn, x_test, y_test, plt.cm.Reds)
    print("K-NN", acc_knn)


def svc(x_train, y_train, x_test, y_test):
    svc = SVC(kernel='linear')
    svc.fit(x_train, y_train)
    svc_pred = svc.predict(x_test)
    acc_svc = accuracy_score(y_test, svc_pred)
    cm_svc = confusion_matrix(y_test, svc_pred)
    print("SVM", acc_svc)
    title = "SVM Confussion Matrix"
    visualize_confussion_matrix(title, svc, x_test, y_test, plt.cm.Oranges)


def gaussian_naive_bayes(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    gnb_pred = gnb.predict(x_test)
    acc_gnb = accuracy_score(y_test, gnb_pred)
    cm_gnb = confusion_matrix(y_test, gnb_pred)
    print("Gaussian Naive Bayes", acc_gnb)
    title = "Gaussian Naive Bayes Confussion Matrix"
    visualize_confussion_matrix(title, gnb, x_test, y_test, plt.cm.YlOrBr)


def neural_network_model(x_train, y_train, x_test, y_test):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation='relu', input_dim=x_train.shape[1]))
    classifier.add(Dense(units=6, activation='relu'))
    classifier.add(Dense(units=3, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(x_train, y_train, epochs=50)
    k_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_test.argmax(axis=1), k_pred.argmax(axis=1))
    print(cm)
    plt.title("Neural Network Confussion Matrix")
    sns.heatmap(cm, annot=True, fmt='.1f')
    plt.show()


def run_with_k_fold_techniques(samples, y_sample):
    print("run_with_k_fold_techniques START")
    x_train, x_test, y_train, y_test = k_fold_cross_validation(samples, y_sample)
    logistic_regression(x_train, y_train, x_test, y_test)
    knn(x_train, y_train, x_test, y_test)
    svc(x_train, y_train, x_test, y_test)
    gaussian_naive_bayes(x_train, y_train, x_test, y_test)
    decision_tree(x_train, y_train, x_test, y_test)
    random_forest(x_train, y_train, x_test, y_test)

    # Neural Network Multi Class Classification
    # Convert categorical variable into dummy/indicator variables
    Y = pd.get_dummies(data['winner'])
    # Send categorical variable to K fold
    x_train, x_test, y_train, y_test = k_fold_cross_validation(samples, Y.values)
    neural_network_model(x_train, y_train, x_test, y_test)
    print("run_with_k_fold_techniques FINISH")


def run_with_train_test_split_techniques(samples, y_sample):
    print("run_with_train_test_split_techniques START")
    x_train, x_test, y_train, y_test = train_test_split_method(samples,y_sample)
    logistic_regression(x_train, y_train, x_test, y_test)
    knn(x_train, y_train, x_test, y_test)
    svc(x_train, y_train, x_test, y_test)
    gaussian_naive_bayes(x_train, y_train, x_test, y_test)
    decision_tree(x_train, y_train, x_test, y_test)
    random_forest(x_train, y_train, x_test, y_test)

    # Neural Network Multi Class Classification
    # Convert categorical variable into dummy/indicator variables
    Y = pd.get_dummies(data['winner'])
    # Send categorical variable to train-test-split
    x_train, x_test, y_train, y_test = train_test_split_method(samples, Y.values)
    neural_network_model(x_train, y_train, x_test, y_test)
    print("run_with_train_test_split_techniques FINISH")


print("PCA results START")
run_with_k_fold_techniques(pca_obj_df.values, y)
run_with_train_test_split_techniques(pca_obj_df,y)
print("PCA results FINISH")



