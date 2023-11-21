import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read the SMS data from a CSV file into a DataFrame
sms=pd.read_csv('spam.csv', encoding='latin-1')

sms=sms.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
sms=sms.rename(columns={"v1":"label","v2":"text"})

# Count and print the number of 'ham' and 'spam' labels in the 'label' column
sms.label.value_counts()

# Generate and display basic statistics about the DataFrame
sms.describe()

# Create a new column 'length' to store the character count of each SMS message
sms['length']=sms['text'].apply(len)

# Display the DataFrame with the new 'length' column
print(sms)

# Create histograms to visualize the distribution of message lengths for 'ham' and 'spam' labels
sms.hist(column='length', by='label', bins=50, figsize=(15, 8))

# Map 'ham' to 0 and 'spam' to 1 in the 'label' column for numerical encoding
sms['label'] = sms['label'].map({'ham': 0, 'spam': 1})

# Add labels to the x and y-axis
plt.xlabel('Length of Message')
plt.ylabel('Frequency')
plt.show()

# Initialize a CountVectorizer for text feature extraction
vectorizer =CountVectorizer()

# Fit the CountVectorizer to the text data, transforming it into a matrix of word counts
text=vectorizer.fit_transform(sms['text'])

# Split the data into training and testing sets (features and labels)
x_train, x_test, y_train, y_test= train_test_split(text, sms['label'], test_size=0.20, random_state=1)

# Hyperparameter tuning using GridSearchCV, alpha
param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
}

# MNB model for the classification task
model = MultinomialNB()

# Initialize the GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # number of cross-validation
    n_jobs=-1  # use all available cores
)

# Fit the grid search model
grid_search.fit(x_train, y_train)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use the best estimator for predictions
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(x_test)

# Sample input text to be tested
sample_inputs = [
    "Congratulations! You've won a $1000 gift card. Click the link to claim your prize now! Don't miss out on this amazing offer.",
    "Hi, I hope you're doing well. Just wanted to check in and see if you're available for a quick call later. Let me know when it's a good time for you.",
    "URGENT: You've been selected to win a luxury vacation for two! Claim your prize now.",
    "Hey, it's been a while since we caught up. How's everything going on your end?",
    "Meet singles in your area tonight. Don't miss this opportunity!",
    "Dear valued customer, your account needs verification. Click the link to confirm your details.",
    "Congratulations on your recent achievement! You're doing a great job.",
    "Your package will be delivered tomorrow. Please ensure someone is available to receive it.",
    "You've won a free pass to the upcoming conference. Register now to secure your spot.",
    "Happy birthday! We wish you a day filled with joy and laughter.",
    "Please find attached the report you requested. Let me know if you need any further information.",
    "Your appointment is confirmed for next Tuesday at 2 PM. We look forward to seeing you."
]

# Transform the sample input texts
sample_inputs_transformed = vectorizer.transform(sample_inputs)

# Make predictions on the sample inputs
predictions = best_model.predict(sample_inputs_transformed)

# Interpret the predictions
for i, prediction in enumerate(predictions):
    if prediction == 0:
        print(f"Sample input {i + 1} is predicted as 'ham' (not spam).")
    else:
        print(f"Sample input {i + 1} is predicted as 'spam'.")


# Calculate accuracy, precision, recall, and F1 score for the entire test set
accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)

print("--------------------------------")
print("Multinomial NB Metrics on Test Set:")
print("Accuracy score: {:.2f}".format(accuracy))
print("Precision score: {:.2f}".format(precision))
print("Recall score: {:.2f}".format(recall))
print("F1 score: {:.2f}".format(f1))
print("--------------------------------")

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, test_predictions)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

