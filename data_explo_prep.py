#-----TASK 1 - DATA EXPLORATION & PREPARATION----#

import pandas as pd

# Load dataset
df = pd.read_excel(r"D:\Shweta\NextHikes_Projects\PROJECT_7\twitter_disaster.xlsx")

# Quick look
print(df.head())
print(df.info())

# Missing values
print(df.isnull().sum())

# Class distribution
print(df['target'].value_counts())


#---------VISUALISE CLASS DISTRIBUTION------#

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='target', data=df)
plt.title("Disaster vs Non-Disaster Tweets")
plt.show()


#----------Keyword Frequency Analysis-------#

from collections import Counter
import re

# Simple tokenizer
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Separate disaster and non-disaster tweets
disaster_tweets = df[df['target']==1]['text']
non_disaster_tweets = df[df['target']==0]['text']

# Count words
disaster_words = Counter([word for tweet in disaster_tweets for word in tokenize(tweet)])
non_disaster_words = Counter([word for tweet in non_disaster_tweets for word in tokenize(tweet)])

print("Top disaster words:", disaster_words.most_common(10))
print("Top non-disaster words:", non_disaster_words.most_common(10))


#----Handle Missing Values-----#

df['keyword'] = df['keyword'].fillna("unknown")
df['location'] = df['location'].fillna("unknown")


#------Text Cleaning------#

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)


#----------Feature Engineering-----------#

df['keyword_flag'] = df['keyword'].apply(lambda x: 0 if x=="unknown" else 1)
df['text_length'] = df['clean_text'].apply(lambda x: len(x.split()))


#------Train-Test Split----------#

from sklearn.model_selection import train_test_split

X = df[['clean_text','keyword_flag','text_length']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#------Vectorisation-------#

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train['clean_text'])
X_test_tfidf = tfidf.transform(X_test['clean_text'])


#------Combine Features-----#

import scipy.sparse as sp

X_train_final = sp.hstack([X_train_tfidf, 
                           X_train[['keyword_flag','text_length']].values])
X_test_final = sp.hstack([X_test_tfidf, 
                          X_test[['keyword_flag','text_length']].values])



#--------Sentiment Analysis---#

from textblob import TextBlob

df['sentiment'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)


#----Extra Features-----#

df['has_hashtag'] = df['text'].apply(lambda x: 1 if '#' in x else 0)
df['has_mention'] = df['text'].apply(lambda x: 1 if '@' in x else 0)


#----------------------------------------------------------------------------------------#


#-----TASK 2 - MODEL SELECTION & TRAINING----------#

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Baseline models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("Baseline cross-validation scores:")
for name, model in models.items():
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f}")

# Hyperparameter tuning for Logistic Regression
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_tfidf, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV score:", grid.best_score_)


#-----TASK 3 - MODEL EVALUATION & VALIDATION-----#

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Train best Logistic Regression model with tuned params
best_model = LogisticRegression(
    max_iter=1000,
    C=grid.best_params_['C'],
    solver=grid.best_params_['solver']
)
best_model.fit(X_train_tfidf, y_train)
y_pred = best_model.predict(X_test_tfidf)

# Metrics
print("\nEvaluation on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Disaster','Disaster'],
            yticklabels=['Non-Disaster','Disaster'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_prob = best_model.predict_proba(X_test_tfidf)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------#

#-----TASK 4 - MODEL DEPLOYMENT WITH WEB INTERFACE---------#

#----Serialize the Model-----#

import pickle

# Save model
with open("disaster_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save TF-IDF vectorizer too (needed for preprocessing)
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)


