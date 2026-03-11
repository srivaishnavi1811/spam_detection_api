import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split features and labels
X = data["message"]
y = data["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ML pipeline
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model/spam_model.pkl")

print("Model trained and saved successfully!")