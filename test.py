"""
Является ли сообщение положительным или отрицательным
0=positive 1=Negative
"""
import re
import nltk
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords', quiet=True)

movie_data = pd.read_csv(r"E:\学习\信息处理\数据\spam_or_not_spam.csv")
x, y = movie_data.email, movie_data.label

documents = []

Quantity = WordNetLemmatizer()

for sen in range(0, len(x)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(x[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [Quantity.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
x = tfidfconverter.fit_transform(documents).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("0=positive 1=Negative")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier, picklefile)
with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

# y_pred2 = model.predict(x_test)

# print(confusion_matrix(y_test, y_pred2))
# print(classification_report(y_test, y_pred2))
# print(accuracy_score(y_test, y_pred2))
