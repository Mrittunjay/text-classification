"""
SPAM/HAM CLASSIFICATION USING TFIDF-VECTORIZER AND LINEAR-SVC
Author: Sipan Pal
Date: 01-01-2026

DESCRIPTION:
1. Performed data cleaning, reduced words to their root form 
using WordNetLemmatizer
2. Visualized the data using WordCloud to check for most common
words in spam emals
3. Used LabelEncoder to convert the labels into numerical values
4. Then used TfidfVectorizer with character n-grams which helped in
bypassing cleaver word manupulations by spammers.
5. Using PCA and t-SNE visualized higher dimentional tfidf data 
to check if data was linearly separable.
6. Trained linearSVC model and evaluated model efficiency. 

DATA SOURCE:
https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification/data
"""

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA   # Principal Component Analysis
from sklearn.manifold import TSNE       # t-distributed Sthocastic Neighbour Embedding
from sklearn.decomposition import TruncatedSVD

# initialize lemmatizer
nltk.download('punkt')  # Unsupervised trainable sentence tokenizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    # Root words: e.g., 'wins', 'winner', 'winning' will become 'win'
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])

df = pd.read_csv("datasets/shantanudhakadd/email-spam-detection-dataset-classification/versions/1/spam.csv",
                 encoding='latin1')

"""
Using df_copy = df[['v1', 'v2']]:
In Pandas 3.0, the new Copy-on-Write rules are applied.
Behind the scenes, pandas 3.0 initially creates a "lazy" view to save memory and time. 
It does not actually duplicate the data until you attempt to modify either the original or the copy.
could also use df_copy = df[['v1', 'v2']].copy() ---> it used deep=True by default.
"""
df_copy = df[['v1', 'v2']].copy()
print()
print(df_copy.head(5))

df_copy.rename(columns={'v1':'label', 'v2':'email'}, inplace=True)

# Applying the changes to the data before fit_transform
df_copy['email_clean'] = df_copy['email'].apply(lemmatize_text)

print()
print(df_copy.head(5))

# Visulaizing wordCloud
spam_words = " ".join(list(df_copy[df_copy['label'] == 'spam']['email']))
wordcloud = WordCloud(width=600, height=400).generate(spam_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Word Cloud")
plt.show()

# Visulaizing wordCloud after applying lemmatize on data
spam_words_clean = " ".join(list(df_copy[df_copy['label'] == 'spam']['email_clean']))
word_cloud_clean = WordCloud(width=600, height=400).generate(spam_words_clean)
plt.imshow(word_cloud_clean)
plt.axis('off')
plt.title("Word Cloud after lemmatize")
plt.show()


# Converting the labels into numerical values
le = LabelEncoder()
df_copy['label_num'] = le.fit_transform(df_copy['label'])
# ham = 0, spam = 1 : Usually sorted alphabetically
print()
print(le.classes_)
print()

# Splitting data into train test set
X = df_copy['email_clean']
y = df_copy['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vactorizing the data 
# tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True)    # Getting better results
# tfidf = TfidfVectorizer(    # THIS ONE IS WORSE THAN DEFAULT TFIDF WITH SUBLINEAR_TF
#     stop_words='english',
#     sublinear_tf=True,      # Applies logarithmic scaling to the term frequency
#     ngram_range=(1, 2),     # Capture word pairs 
#     min_df = 2,             # Ignore rarewords or typos
#     max_df = 0.7            # Ignore very common words
#     # max_features=10000      # Keeps only the top 10K frequent features 
# )

# Best Result:
"""
Character n-grams will help bypass cleaverly designed words.
1. Spammers often use variation of words like W_inner or W1nner or w-i-n-n-e-r
to bypass word based filters.
    --> A word vectorizer will see these words ans completely unknown seperate words
    --> But a character vectorizer will see the 3-gram inn, nne and ner in all the 
    words, which help the model recognize better patters. 
"""
tfidf = TfidfVectorizer(
            # stop_words='english', # As analyzer is char not word.
            analyzer='char',
            ngram_range=(3, 5)
        )
# Applying tfidf
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Checking data linerity (to figure out if LinearSVC is a suitable model for this problem or not)
X_tfidf_pca = tfidf.fit_transform(df_copy['email_clean']).toarray()
y_pca = df_copy['label_num']    # labelEncoded data (ham:0, spam:1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf_pca)
# Plotting the results
plt.figure(figsize=(10, 6))
for label, color in zip([0, 1], ['blue', 'red']):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                label='Ham' if label == 0 else 'Spam', alpha=0.5, c=color)
plt.title("PCA Projection: Checking for linear separability")
plt.legend()
plt.show()

# Using t-SNE(t-distributed Stocastic Neighbour Embedding) to check linearly separable.
# 1. Sparse-friendly PCA (TruncatedSDV) to reduce noise first
# Pre reduce dimentions to 50 using TruncatedSDV (Standard for sparce text)
sdv = TruncatedSVD(n_components=50, random_state=42)
X_reduced = sdv.fit_transform(tfidf.fit_transform(df_copy['email_clean']))

# 2. Run t-SNE (Set perplexity between 30 and 50 for this dataset size)
tsne = TSNE(n_components=2, perplexity=35, random_state=50, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_reduced)

# 3. Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[y_pca == 0, 0], X_tsne[y_pca == 0, 1], c='blue', label='Ham', alpha=0.6)
plt.scatter(X_tsne[y_pca == 1, 0], X_tsne[y_pca == 1, 1], c='red', label='Spam', alpha=0.6)
plt.title("t-SNE Visualization of Email data")
plt.legend()
plt.show()


# Model training and evaluation
model_lsvc = LinearSVC(class_weight='balanced', random_state=50)
model_lsvc.fit(X_train_tfidf, y_train)

y_pred = model_lsvc.predict(X_test_tfidf)

print("\n----Confusiton Matrix LinearSVC----")
print(confusion_matrix(y_test, y_pred))
print("\n----Classification Report----")
print(classification_report(y_test, y_pred, target_names=le.classes_))