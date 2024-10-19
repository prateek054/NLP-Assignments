import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load data, specifying the encoding as 'latin-1' and using delimiter='\t'
positive_reviews = pd.read_csv('rt-polarity.pos', header=None, names=['text'], encoding='latin-1', delimiter='\t')
negative_reviews = pd.read_csv('rt-polarity.neg', header=None, names=['text'], encoding='latin-1', delimiter='\t')


# Split data
train_pos = positive_reviews.iloc[:4000]
train_neg = negative_reviews.iloc[:4000]
val_pos = positive_reviews.iloc[4000:4500]
val_neg = negative_reviews.iloc[4000:4500]
test_pos = positive_reviews.iloc[4500:]
test_neg = negative_reviews.iloc[4500:]

train_data = pd.concat([train_pos, train_neg])
val_data = pd.concat([val_pos, val_neg])
test_data = pd.concat([test_pos, test_neg])

train_labels = [1] * 4000 + [0] * 4000
val_labels = [1] * 500 + [0] * 500
test_labels = [1] * 831 + [0] * 831

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['text'])
X_val = vectorizer.transform(val_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, train_labels)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))
