# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the spam dataset from the CSV file and convert text labels into numerical form.
2. Transform the message text into numerical features using TF-IDF vectorization.
3. Train an SVM classifier on the training data and make predictions on test data.
4. Evaluate the model performance and classify new incoming messages as spam or ham.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Joshua Abraham Philip
RegisterNumber: 25013744

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("spam.csv (1)")

df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['v2'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_message = ["Congratulations! You have won a free ticket to Bahamas. Call now!"]
new_message_vect = vectorizer.transform(new_message)
prediction = svm_model.predict(new_message_vect)
print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")


*/
```

## Output:
<img width="397" height="197" alt="image" src="https://github.com/user-attachments/assets/947f2588-8170-4feb-8a7c-28482f54a91e" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
