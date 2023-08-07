import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Set the custom data path for NLTK
custom_data_path = "libary"
nltk.data.path.append(custom_data_path)

# Download the 'punkt' resource if not already downloaded
nltk.download('punkt', download_dir=custom_data_path)

# Read the labeled dataset (assuming a CSV file with 'text' and 'category' columns)
data = pd.read_csv("path/to/your/dataset.csv")

# Preprocess text data
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['category'], test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a machine learning model (e.g., Multinomial Naive Bayes)
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)

