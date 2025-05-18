# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb

# Load Data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'cryptonews_preprocessing', 'preprocessed_cryptonews.csv')

# Read CSV file
df = pd.read_csv(file_path)

# Prepare Features and Target
X_text = df['text_clean']              # Cleaned text data
y = df['sentiment_encoded']            # Encoded labels: 0=Negative, 1=Neutral, 2=Positive

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(X_text).toarray()  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Define and Train Models
print("\n[INFO] Training models with SMOTE...\n")

# Initialize models
models = {
    "LGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(max_iter=1000, random_state=42)
}

results = []

for name, model in models.items():
    print(f"[TRAIN] {name}")
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1 Score': f1
    })
    
    print(f"   Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}\n")

# Convert results to DataFrame for comparison 
results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
print("\n[RESULTS] Model Comparison (F1 Score):")
print(results_df)

# Evaluate Best Model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

print(f"\n[REPORT] Classification Report - {best_model_name}:")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.show()