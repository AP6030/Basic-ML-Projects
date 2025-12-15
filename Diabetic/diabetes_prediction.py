import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import pickle
import os

# Load saved model and preprocessing tools
with open('diabetes_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
imputer = model_data['imputer']
columns = model_data['columns']

# Initialize or load prediction history
history_file = 'prediction_history.csv'
if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Prediction'
    ])
    history_df.to_csv(history_file, index=False)

def predict_diabetes():
    try:
        # Get input values from entries
        values = [
            float(entry_pregnancies.get()),
            float(entry_glucose.get()),
            float(entry_bp.get()),
            float(entry_skin.get()),
            float(entry_insulin.get()),
            float(entry_bmi.get()),
            float(entry_dpf.get()),
            float(entry_age.get())
        ]

        # Create DataFrame
        input_df = pd.DataFrame([values], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])

        # Impute missing
        input_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
            input_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
        input_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
            imputer.transform(input_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

        # Feature engineering
        input_df['Glucose_Insulin'] = input_df['Glucose'] * input_df['Insulin']
        model_input_df = input_df.drop(columns=['SkinThickness', 'Insulin', 'BloodPressure'])  # match model features

        # Align columns
        model_input_df = model_input_df[columns]

        # Scale input
        std_data = scaler.transform(model_input_df)

        # Predict
        prediction = model.predict(std_data)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        messagebox.showinfo("Prediction", f"The person is likely: {result}")
        
        # Save prediction to history
        new_row = input_df.iloc[0].to_dict()
        new_row['Prediction'] = result
        global history_df
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        history_df.to_csv(history_file, index=False)
        
        # Update the history table
        update_history_table()

    except Exception as e:
        messagebox.showerror("Error", str(e))

def update_history_table():
    # Clear existing data
    for item in history_tree.get_children():
        history_tree.delete(item)
    
    # Add data to treeview
    for i, row in history_df.iterrows():
        values = [row['Pregnancies'], row['Glucose'], row['BloodPressure'], 
                 row['SkinThickness'], row['Insulin'], row['BMI'], 
                 row['DiabetesPedigreeFunction'], row['Age'], row['Prediction']]
        history_tree.insert("", "end", values=values)

# Tkinter GUI
root = tk.Tk()
root.title("Diabetes Predictor")
root.geometry("900x600")

# Create notebook (tabs)
nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True, padx=10, pady=10)

# Input Frame
input_frame = ttk.Frame(nb)
nb.add(input_frame, text="Prediction Input")

# History Frame
history_frame = ttk.Frame(nb)
nb.add(history_frame, text="Prediction History")

# Input labels and entries
labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
entries = []

for i, label in enumerate(labels):
    tk.Label(input_frame, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='e')
    entry = tk.Entry(input_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

entry_pregnancies, entry_glucose, entry_bp, entry_skin, entry_insulin, entry_bmi, entry_dpf, entry_age = entries

# Predict button
tk.Button(input_frame, text="Predict", command=predict_diabetes, bg='green', fg='white').grid(row=9, column=0, columnspan=2, pady=20)

# History table
history_tree = ttk.Treeview(history_frame)
history_tree["columns"] = ("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Prediction")

# Configure columns
history_tree.column("#0", width=0, stretch=tk.NO)
for col in history_tree["columns"]:
    history_tree.column(col, anchor=tk.CENTER, width=90)
    history_tree.heading(col, text=col)

# Add scrollbar
scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=history_tree.yview)
history_tree.configure(yscrollcommand=scrollbar.set)

# Pack history components
history_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
scrollbar.pack(side="right", fill="y", pady=10)

# Initialize history table
update_history_table()

root.mainloop()
