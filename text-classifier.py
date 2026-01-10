from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

"""
Bayes Theorem:
Posterior = (Prior * Likelihood)/Evidence
P(Y|X) = P(Y) * P(X|Y) / P(X)
"""

# Small data for dummy model training
texts = [
    "I love programming.", "Python is amaizing.",
    "I enjoy machine learning.", "The weather is nice today.", "I like algo.",
    "Machine learning is fascinating.", "Natural Language Processing is a part of AI.",
    "I Want to travel Japan at least once in my life.", "I wanted to experience the European, Australean, Japanese and US work culture.",
    "It is tough to get a good job in AI based tech industries now a days.", 
    "AI, ML and Gen AI are the booming technologies of the present.",
    "I tried out my old passion for modeling.", "I could not yet try my passion in acting.",
    "I should have tried all these experiments with life during my early age, but now it is too late."
]

labels = [
    "tech", "tech", "tech", "non-tech", "tech", "tech", "tech", "non-tech",
    "non-tech", "tech", "tech", "non-tech", "non-tech", "non-tech"
]

# Converting text to numerical data
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(texts)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

# Training the Naive Bayse Classifier(MultinominalNB classifier model)
print("Model Training in Progress . . . ")
model = MultinomialNB()
model.fit(X_train, y_train)

# Making prediction on test set
y_pred = model.predict(X_test)

# Evaluating the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Saving the model
model_name = 'my_naive_bayse_v1.pkl'
vectorizer_name = 'count_vectorizer.pkl'
joblib.dump(model, model_name)
joblib.dump(vectorizer, vectorizer_name)
print(f"Saved trained model as : {model_name}")
print(f"Saved vectorizer as {vectorizer_name}")

# Loading the saved model
model_v1 = joblib.load(model_name)
vectorizer_v1 = joblib.load(vectorizer_name)

# Running predction on input data
input_data = input("Enter any text to classify as tech or non-tech text: \n")
print(f"Input data: {input_data}")
new_text_vectorized = vectorizer_v1.transform([input_data])
# Running predction
predction = model_v1.predict(new_text_vectorized)

print("Prediction class:", predction)


"""
The model is not a generic model, it only predicts properly 
with the words it has seen during the training

Naive Bayes + CountVectorizer:
After the vectorization, the model does not understand meaning.
It only sees word counts and asks:
"Which class has seen these words more often during training?"
"""

# NEXT TASK: Use TFIDF and LinearSVC for spam/ham classification
