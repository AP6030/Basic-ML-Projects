import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import os

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# checking the distribution of Target Variable
# 1 -> Benign
# 0 -> Malignant
print(data_frame['label'].value_counts())

# Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Splitting the data into training data & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# setting up the layers of Neural Network
tf.random.set_seed(3)
model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(30,)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])

# compiling the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the Neural Network
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# --- VISUALIZATION (Optional, can be commented out if not needed) ---
# Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Loss Graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# --------------------------------------------------------------------


# Making a Predictive System
def make_prediction(input_data):
    # change the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    input_data_std = scaler.transform(input_data_reshaped)

    prediction = model.predict(input_data_std)
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    if predicted_class == 0:
        return "The tumor is Malignant (Cancerous)"
    else:
        return "The tumor is Benign (Non-Cancerous)"

# Example prediction
input_data = (17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189)
result = make_prediction(input_data)
print(f"Prediction result: {result}")


# ===================================================================
# UPDATED SECTION: Save the model and scaler to the specified path
# ===================================================================

# Define the directory path
# Using a raw string (r'...') to handle backslashes in Windows paths correctly
save_path = r'C:\Users\ASUS\Intern-Pe-Updated\Breast-Cancer'

# Create the target directory if it does not exist
os.makedirs(save_path, exist_ok=True)

# Define full file paths
model_file_path = os.path.join(save_path, 'breast_cancer_model.pkl')
scaler_file_path = os.path.join(save_path, 'breast_cancer_scaler.pkl')

# Save the trained model
with open(model_file_path, 'wb') as f:
    pickle.dump(model, f)

# Save the scaler object
with open(scaler_file_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nModel successfully saved to: {model_file_path}")
print(f"Scaler successfully saved to: {scaler_file_path}")