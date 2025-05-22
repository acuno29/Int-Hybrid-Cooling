import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
# part na to i reread ni python yung CSV file na dataset
data_file = 'temperature_humidity_data.csv'
df = pd.read_csv(data_file)

# Preview data
# makikita sa console yung dataset file
print("Initial data preview:")
print(df.head())

# Ensure numeric conversion for temperature and humidity, coercing errors to NaN
# set a parameters lang dito to read correct values Decimals,Zero,Positive,Negative or Blank
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')

# Drop rows with NaN in key columns
# mareremove dito yung mga blank cells 
df.dropna(subset=['Temperature', 'Humidity'], inplace=True)

# Temperature labeling with extended classes
# Conditions ng Temperature no negative temperature kase nasa pilipinas tayo tropical weather
def label_temperature(temp):
    if pd.isna(temp):
        return 'unknown'
    elif temp <= 10:
        return 'very cold'
    elif 11 <= temp <= 16:
        return 'cold'
    elif 17 <= temp <= 22:
        return 'cool'
    elif 23 <= temp <= 28:
        return 'warm'
    elif 29 <= temp <= 34:
        return 'hot'
    elif temp > 34:
        return 'very hot'
    else:
        return 'unknown'

# Humidity labeling with multiple classes
# Conditions ng Humidity
def label_humidity(hum):
    if pd.isna(hum):
        return 'unknown'
    elif hum <= 30:
        return 'dry'
    elif 31 <= hum <= 60:
        return 'comfortable'
    elif 61 <= hum <= 80:
        return 'humid'
    elif hum > 80:
        return 'very humid'
    else:
        return 'unknown'

# Additional sensor action labeling for fan (based on temp)
# Conditions ng Fan if anong action
def label_fan_action(temp):
    if pd.isna(temp):
        return 'unknown'
    elif temp <= 10:
        return 'off'
    elif 11 <= temp <= 22:
        return 'low'
    elif 23 <= temp <= 28:
        return 'medium'
    elif 29 <= temp <= 34:
        return 'high'
    elif temp > 34:
        return 'max'
    else:
        return 'unknown'

# Additional sensor action labeling for humidifier (based on humidity)
# Condition ng actions ng humidifier 
def label_humidifier_action(hum):
    if pd.isna(hum):
        return 'unknown'
    elif hum <= 30:
        return 'off'
    elif 31 <= hum <= 60:
        return 'low'
    elif 61 <= hum <= 80:
        return 'medium'
    elif hum > 80:
        return 'high'
    else:
        return 'unknown'

# Apply labels
# print data sa added column 
df['Temp_Label'] = df['Temperature'].apply(label_temperature)
df['Humidity_Label'] = df['Humidity'].apply(label_humidity)
df['Fan_Action'] = df['Temperature'].apply(label_fan_action)
df['Humidifier_Action'] = df['Humidity'].apply(label_humidifier_action)

print("\nLabeled data sample:")
print(df[['Date', 'Day', 'Time', 'Temperature', 'Temp_Label', 'Humidity', 'Humidity_Label', 'Fan_Action', 'Humidifier_Action']].head(10))

# Features and target for classification (temperature label)
# gagawa ng dalawang types ng label x at y
X = df[['Temperature', 'Humidity']]
y = df['Temp_Label']

# Ensure at least two classes exist
if y.nunique() < 2:
    print("Not enough classes in temperature labels for training.")
else:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    # dito mag sstart mag train yung SVM algo 
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Predict and evaluate
    # lalabas dito yung predictions and accurracy
    y_pred = model.predict(X_test)
    print("\nClassification Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save labeled dataset to CSV
# then dito last step i sasave nya na sa panibagong csv file yun results nyo sa same folder
output_file = 'labeled_temperature_humidity_data.csv'
df.to_csv(output_file, index=False)
print(f"\nLabeled dataset saved to {output_file}")

