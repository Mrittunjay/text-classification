Working with conventional ML models:
"text-classifier-v3-tfidf-linearSVM.py"

SPAM/HAM CLASSIFICATION USING TFIDF-VECTORIZER AND LINEAR-SVC

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


"text-classifier-v2.py"
SPAM/HAM CLASSIFICATION INITIAL RUN


"text-classifier.py"
DUMMY MODEL PRACTICE WITH SMALL DATA (TOPIC CLASSIFICATION PROBLEM)
Model used: MultinomialNB  --> Naive Bayes
Vectorization: CountVectorizer(BoW)
