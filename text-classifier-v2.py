# Use TFIDF and LinearSVC for spam/ham classification

#---------------------RUN ONCE-------------------------------------------------------
# # Download email spam detection data from kaggle
# import os
# import kagglehub

# # Default kagglehub cache location is "C:\Users\DELL\.cache\kagglehub\datasets"
# # To set custom kagglehub cache location
# os.environ["KAGGLEHUB_CACHE"] = "D:/projects/deep-learning-practice/gen-AI/text-classifier-using-tfidf-with-linearSVM"

# # help(kagglehub.dataset_download)
# path = kagglehub.dataset_download(
#     handle="shantanudhakadd/email-spam-detection-dataset-classification")
# print(f"DATASET PATH: {path}")
#------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("datasets/shantanudhakadd/email-spam-detection-dataset-classification/versions/1/spam.csv",
                 encoding='latin1')
print(df.head(5))
print(df.shape)
print(df.info())
print(df.isnull().sum())
print()
print(df['Unnamed: 3'].unique())
print()

# Keeping useful columns and discurding other columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.rename(columns={'v1':'label', 'v2':'email'}, inplace=True)
df = df[['email', 'label']] # Changing column position
print(df.head(5))
print()

# PERFORMING EDA
# 1. Class distribution
print("Class Distribution: \n", df['label'].value_counts())

# 2. Visualizing class distribution
sns.countplot(x='label', data=df)
plt.title('Spam vs ham count')
plt.show()

# 3. Message length intution
df['length'] = df['email'].apply(len)
df.hist(column='length', by='label', bins=50, figsize=(12, 4))
plt.suptitle("Message length visulization")
plt.show()

# 4. Wordcloud intution: What word define spam?
spam_words = ' '.join(list(df[df['label'] == 'spam']['email']))
wordcloud = WordCloud(width=600, height=400).generate(spam_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Most common words in spam emails")
plt.show()


# DATA PREPARATION:

# 1. Converting the labels into numeric values 
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])
# ham = 0, spam = 1 (usually sorted alphabetically)
print(f"Classes: {le.classes_}")

# 2. Split data
X = df['email']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# 3. Vactorization of the data using TF-IDF
# # This implementation of TFIDF uses word level features.
# tfidf = TfidfVectorizer(stop_words='english', sublinear_tf=True)

# Fine tuning TF-IDF
# Using Bi-grams/Tri-grams to better identify spamming word payers
# setting min_df=2 ignores the words that apper only once (likely typos)
# max_df=0.7 to ignore words that appear in 70% of emails (Common words that does not help in classification)
tfidf = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,  # Applies logarithmic scaling to the term frequency, dempens the effect of very high frequency words
    ngram_range=(1, 2), # Capture word pairs
    min_df = 2 ,        # Ignores rare words/Typos
    max_df = 0.7        # Ignores overly common words
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# MODEL TRAINING AND EVALUATION

# 1. Modeling with imbalance handling
# class_weight = 'balanced' gives more importance to the 'Spam' minority class
model_lsvc = LinearSVC(class_weight='balanced', random_state=50)
model_lsvc.fit(X_train_tfidf, y_train)

# 2. Evaluation
y_pred = model_lsvc.predict(X_test_tfidf)

"""
                    Predicted: HAM	       Predicted: SPAM
Actual: HAM	     952 (True Negatives)	 2 (False Positives)
Actual: SPAM	 13 (False Negatives)	 148 (True Positives)

952 (True Negatives): Correct! 952 real emails stayed in the inbox.
148 (True Positives): Correct! 148 junk emails were successfully caught.
2 (False Positives): Bad mistake! 2 important emails were wrongly labeled as spam and sent to the junk folder.
13 (False Negatives): Minor annoyance. 13 spam emails "slipped through" and appeared in the primary inbox. 
"""
print("\n----Confusion Matrix LinarSVC----")
print(confusion_matrix(y_test, y_pred))

"""
1. Precision: "When I say it's Spam, how often am I right?"
Spam (0.99): Of all emails labeled "Spam," 99% were actually spam. This is extremely high; your model rarely sends a good email to the junk folder. 

2. Recall: "Of all the real Spam out there, how many did I catch?"
Spam (0.92): Your model caught 92% of all total spam messages. It missed about 8% (those 13 emails in the inbox). 

3. F1-Score: The Balance
This is the "average" of Precision and Recall. Since you want to catch as much spam as possible (Recall) without blocking real emails (Precision), a 0.95 for Spam is excellent. 

4. Support: The Reality Check
Support is just the count of actual emails in each group. Your test set had 954 Ham and 161 Spam. This confirms your data is imbalanced—you have much more "normal" mail than junk. 

5. Macro vs. Weighted Average
Macro Avg (0.97): This treats Spam and Ham as equally important. It’s like a teacher giving the same weight to a 5-student class as a 50-student class.
Weighted Avg (0.99): This gives more weight to the bigger group (Ham). Because you are so good at identifying Ham, this number is higher. 
"""
print("\n----Classification Report LinearSVC----")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# Using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize with balanced weights to penelize spam misclassification more heavily
rfc_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
rfc_model.fit(X_train_tfidf, y_train)

# Prediction and Evaluation
y_pred_rfc = rfc_model.predict(X_test_tfidf)

print("\n----Confusion Matrix RFC----")
print(confusion_matrix(y_test, y_pred_rfc))
print("\n----RandomForest Classifier Report----")
print(classification_report(y_test, y_pred_rfc, target_names=le.classes_))
"""
----CONFUSION MATRIX RFC---- (sklearn standard confusion matrix)
             predicted ham      predicted spam
Actual ham      [[954(TN)              0(FP)]
Actual spam     [ 29(FN)            132(TP)]]

Performance metrices:
Precision: TP / TP + FP
Recall: TP / TP + FN
misclassified 29 spam emails as ham

                    Predicted: Positive	Predicted: Negative
Actual: Positive	True Positive (TP)	False Negative (FN)
Actual: Negative	False Positive (FP)	True Negative (TN)
"""

# # USING GRIDSEARCH-CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# from sklearn.model_selection import GridSearchCV
# params_grid = {
#     'n_estimators': [100, 150, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# # GridSearchCV uses StratifiedKFold internally for classification
# # refit=True is default, it retrains the model on the whole training set
# grid_search = GridSearchCV(rfc_model, params_grid, cv=5, scoring='recall', refit=True)
# grid_search.fit(X_train_tfidf, y_train)

# print(f"\n Best Parameters from GridSearchCV: {grid_search.best_params_}")
# y_pred_GSCV = grid_search.predict(X_test_tfidf)

# print("\n----Confusion Metrix GridSearchCV----")
# print(confusion_matrix(y_test, y_pred_GSCV))
# print("\n----Classification Report after Hyperparameter tuning----")
# print(classification_report(y_test, y_pred_GSCV, target_names=le.classes_))


# # USING RANDOM FOREST CLASSAFIER WITH STRATIFIED K FOLD CROSSVALIDATION 
# from sklearn.model_selection import StratifiedKFold, cross_val_score

# # Defining the strategy
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

# # Getting scores for spam Recall (to see how many spam messages we consistantly miss)
# cv_score = cross_val_score(rfc_model, X_train_tfidf, y_train, cv=skf, scoring='recall')

# # On average the Random forest caught __% of all the spam calls across 5 different tests
# print(f"Mean Recall: {cv_score.mean():.4f}")
# # Low standard dev means the score across all 5 folds were very close, which means model is stabel.
# print(f"Recall standard deviation: {cv_score.std():.4f}")


# USING XGBOOST CLASSIFIER:
from xgboost import XGBClassifier

# Calculate imbalance ratio: count(ham) / count(spam)
# Classes: ['ham' 'spam']
ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

# 'scale_pos_weight tells XGBoost to give more importance to the minority class
xgb_model = XGBClassifier(scale_pos_weight=ratio, random_state=50)

xgb_model.fit(X_train_tfidf, y_train)

y_pred_xgb = xgb_model.predict(X_test_tfidf)

print("\n----Confusion Matrix XGB----")
print(confusion_matrix(y_test, y_pred_xgb))
print("\n----Classification Report of XGB----")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))


# USING LightGBM Classifier
from lightgbm import LGBMClassifier

# is_unbalance = True automatically handles the waiting between ham and spam
lgbm_model = LGBMClassifier(is_unbalance=True, random_state=50)
lgbm_model.fit(X_train_tfidf, y_train)

y_pred_lgbm = lgbm_model.predict(X_test_tfidf)

print("\n----Confusion Matrix LightGBMClassifier----")
print(confusion_matrix(y_test, y_pred_lgbm))
print("\n----Classification Report of LightGBMClassifier----")
print(classification_report(y_test, y_pred_lgbm, target_names=le.classes_))


# NEXT TASK: (Advanced) Multi-Layer spam detection pipeline (industry standard)