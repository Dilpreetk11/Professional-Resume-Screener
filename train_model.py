import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def main():
    print("Loading data...")
    df = pd.read_csv('UpdatedResumeDataSet.csv')
    
    print("Cleaning resume texts...")
    df['cleaned_resume'] = df['Resume'].apply(lambda x: cleanResume(x))
    
    print("Encoding categories...")
    le = LabelEncoder()
    df['Category'] = le.fit_transform(df['Category'])
    
    print("Vectorizing contents...")
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
    X = tfidf.fit_transform(df['cleaned_resume'])
    y = df['Category']
    
    # Optional train-test split to train on full but we only need a model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model...")
    # Using KNeighbors with OneVsRest as usually done in this specific project
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy*100:.2f}%")
    
    print("Saving models to disk...")
    pickle.dump(le, open('encoder.pkl', 'wb'))
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
    pickle.dump(clf, open('clf.pkl', 'wb'))
    print("Done!")

if __name__ == "__main__":
    main()
