import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#Hello if ur checking this kindly take notes on these comments the will show the parts by parts functions of da scipt.

# Read the CSV file, skipping unnecessary rows
data_file = 'DATA_temp_hum.csv'
combined_df = pd.read_csv(data_file, skiprows=3)  

print("DataFrame after reading CSV:")
print(combined_df.head())

#Reshape DataFrame to extract temperature and humidity columns
temperature_columns = combined_df.columns[1::2]  
humidity_columns = combined_df.columns[2::2]    

reshaped_data = []
for index, row in combined_df.iterrows():
    for temp_col, hum_col in zip(temperature_columns, humidity_columns):
        if pd.notna(row[temp_col]) and pd.notna(row[hum_col]):
            reshaped_data.append({
                'Time': row[0],  
                'Temperature': row[temp_col],
                'Humidity': row[hum_col]
            })

reshaped_df = pd.DataFrame(reshaped_data)

# Convert to numeric, coercing errors to NaN
reshaped_df['Temperature'] = pd.to_numeric(reshaped_df['Temperature'], errors='coerce')
reshaped_df['Humidity'] = pd.to_numeric(reshaped_df['Humidity'], errors='coerce')

# Drop rows with NaNs
reshaped_df.dropna(subset=['Temperature', 'Humidity'], inplace=True)

# Define labeling functions with your thresholds
def label_fan(temp):
    if pd.isna(temp):
        return 'unknown'
    if temp <= 30:
        return 'normal'
    else:
        return 'hot'

def label_humidifier(hum):
    if pd.isna(hum):
        return 'unknown'
    if hum <= 60:
        return 'normal'
    else:
        return 'humid'

# Apply labeling
reshaped_df['Fan'] = reshaped_df['Temperature'].apply(label_fan)
reshaped_df['Humidifier'] = reshaped_df['Humidity'].apply(label_humidifier)

# Additional sensor action labels
def label_fan_action(temp):
    if pd.isna(temp):
        return 'unknown'
    if 27 <= temp < 29:
        return 'comfortable'
    elif 29 <= temp < 31:
        return 'uncomfortable'
    elif 32 <= temp <= 45:
        return 'unbearable'
    else:
        return 'unknown'

def label_humidifier_action(hum):
    if pd.isna(hum):
        return 'unknown'
    if 1 <= hum <= 60:
        return 'off'
    elif 61 <= hum <= 100:
        return 'on'
    else:
        return 'unknown'

reshaped_df['Fan_Action'] = reshaped_df['Temperature'].apply(label_fan_action)
reshaped_df['Humidifier_Action'] = reshaped_df['Humidity'].apply(label_humidifier_action)

print(reshaped_df[['Temperature', 'Fan', 'Fan_Action', 'Humidity', 'Humidifier', 'Humidifier_Action']].head())

# Features and Labels (choose one target to classify)
X = reshaped_df[['Temperature', 'Humidity']]

# classify fan labels
y = reshaped_df['Fan']

#  Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

#  Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#  Save labeled DataFrame to CSV
output_file = 'labeled_temperature_humidity.csv'
reshaped_df.to_csv(output_file, index=False)
print(f"Labeled data saved to {output_file}")
