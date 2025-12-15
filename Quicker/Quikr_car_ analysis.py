import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import os

# Load the car data
try:
    if os.path.exists('Quicker/Cleaned_Car_data.csv'):
        car_data = pd.read_csv('Quicker/Cleaned_Car_data.csv')
    else:
        car_data = pd.read_csv('Quicker/quikr_car.csv')
        car_data['year'] = car_data['year'].astype(int)
        car_data = car_data[car_data['Price'] != 'Ask For Price']
        car_data['Price'] = car_data['Price'].str.replace(',', '').astype(int)
        car_data['kms_driven'] = car_data['kms_driven'].str.split().str.get(0).str.replace(',', '')
        car_data = car_data[car_data['kms_driven'].str.isnumeric()]
        car_data['kms_driven'] = car_data['kms_driven'].astype(int)
        car_data = car_data[~car_data['fuel_type'].isna()]
        car_data['name'] = car_data['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
        car_data = car_data.reset_index(drop=True)
        # Save cleaned data for future use
        car_data.to_csv('Quicker/Cleaned_Car_data.csv', index=False)
    
    data_loaded = True
    
    # Get unique car names for dropdown
    unique_car_names = sorted(car_data['name'].unique())
    
    # Get min and max prices for range validation
    min_price = car_data['Price'].min()
    max_price = car_data['Price'].max()

except Exception as e:
    messagebox.showerror("Error", f"Error loading or preparing data: {str(e)}")
    data_loaded = False
    unique_car_names = []
    min_price = 0
    max_price = 1000000

# Initialize or load search history
history_file = 'Quicker/car_search_history.csv'
if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(columns=[
        'car_name', 'min_price', 'max_price', 'search_date'
    ])
    history_df.to_csv(history_file, index=False)

def search_cars():
    if not data_loaded:
        messagebox.showerror("Error", "Data not loaded. Please check the data files.")
        return
        
    try:
        # Get input values
        selected_car = car_var.get()
        min_price_val = int(entry_min_price.get()) if entry_min_price.get() else min_price
        max_price_val = int(entry_max_price.get()) if entry_max_price.get() else max_price
        
        # Validate price range
        if min_price_val > max_price_val:
            messagebox.showerror("Error", "Minimum price cannot be greater than maximum price")
            return
        
        # Filter data based on inputs
        filtered_data = car_data.copy()
        
        if selected_car != "All Cars":
            filtered_data = filtered_data[filtered_data['name'] == selected_car]
            
        filtered_data = filtered_data[
            (filtered_data['Price'] >= min_price_val) & 
            (filtered_data['Price'] <= max_price_val)
        ]
        
        # Display results
        update_results_table(filtered_data)
        
        # Save search to history
        new_row = {
            'car_name': selected_car,
            'min_price': min_price_val,
            'max_price': max_price_val,
            'search_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        global history_df
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        history_df.to_csv(history_file, index=False)
        
        # Update the history table
        update_history_table()
        
        # Show summary
        result_count = len(filtered_data)
        if result_count > 0:
            result_label.config(text=f"Found {result_count} cars matching your criteria")
        else:
            result_label.config(text="No cars found matching your criteria")
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

def update_results_table(data):
    # Clear existing data
    for item in results_tree.get_children():
        results_tree.delete(item)
    
    # Add data to treeview
    for i, row in data.iterrows():
        values = [
            row['name'], 
            row['company'], 
            row['year'], 
            row['kms_driven'], 
            row['fuel_type'], 
            f"₹{row['Price']:,.2f}"
        ]
        results_tree.insert("", "end", values=values)

def update_history_table():
    # Clear existing data
    for item in history_tree.get_children():
        history_tree.delete(item)
    
    # Add data to treeview
    for i, row in history_df.iterrows():
        values = [
            row['car_name'], 
            f"₹{row['min_price']:,.2f}", 
            f"₹{row['max_price']:,.2f}", 
            row['search_date']
        ]
        history_tree.insert("", "end", values=values)

# Tkinter GUI
root = tk.Tk()
root.title("Car Search by Price Range")
root.geometry("900x700")

# Create notebook (tabs)
nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True, padx=10, pady=10)

# Search Frame
search_frame = ttk.Frame(nb)
nb.add(search_frame, text="Search Cars")

# History Frame
history_frame = ttk.Frame(nb)
nb.add(history_frame, text="Search History")

# Search inputs
input_frame = ttk.LabelFrame(search_frame, text="Search Criteria")
input_frame.pack(fill="x", padx=10, pady=10)

# Car dropdown
tk.Label(input_frame, text="Car Model").grid(row=0, column=0, padx=10, pady=5, sticky='e')
car_var = tk.StringVar()
car_choices = ["All Cars"] + unique_car_names
car_dropdown = ttk.Combobox(input_frame, textvariable=car_var, values=car_choices, width=30)
car_dropdown.grid(row=0, column=1, padx=10, pady=5)
car_dropdown.current(0)  # Set default value to "All Cars"

# Price range inputs
tk.Label(input_frame, text="Minimum Price (₹)").grid(row=1, column=0, padx=10, pady=5, sticky='e')
entry_min_price = tk.Entry(input_frame, width=30)
entry_min_price.grid(row=1, column=1, padx=10, pady=5)
entry_min_price.insert(0, str(min_price))

tk.Label(input_frame, text="Maximum Price (₹)").grid(row=2, column=0, padx=10, pady=5, sticky='e')
entry_max_price = tk.Entry(input_frame, width=30)
entry_max_price.grid(row=2, column=1, padx=10, pady=5)
entry_max_price.insert(0, str(max_price))

# Search button
search_button = tk.Button(input_frame, text="Search Cars", command=search_cars, 
                         bg='blue', fg='white', font=('Arial', 12, 'bold'))
search_button.grid(row=3, column=0, columnspan=2, pady=10)

# Results label
result_label = tk.Label(search_frame, text="Enter search criteria and click 'Search Cars'", 
                       font=("Arial", 10, "italic"), fg="blue")
result_label.pack(pady=5)

# Results table
results_frame = ttk.LabelFrame(search_frame, text="Search Results")
results_frame.pack(fill="both", expand=True, padx=10, pady=10)

results_tree = ttk.Treeview(results_frame)
results_tree["columns"] = ("Name", "Company", "Year", "KMs Driven", "Fuel Type", "Price")

# Configure columns
results_tree.column("#0", width=0, stretch=tk.NO)
for col in results_tree["columns"]:
    results_tree.column(col, anchor=tk.CENTER, width=140)
    results_tree.heading(col, text=col)

# Add scrollbar for results
results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_tree.yview)
results_tree.configure(yscrollcommand=results_scrollbar.set)

# Pack results components
results_tree.pack(side="left", fill="both", expand=True)
results_scrollbar.pack(side="right", fill="y")

# History table
history_tree = ttk.Treeview(history_frame)
history_tree["columns"] = ("Car Model", "Min Price", "Max Price", "Search Date")

# Configure history columns
history_tree.column("#0", width=0, stretch=tk.NO)
for col in history_tree["columns"]:
    history_tree.column(col, anchor=tk.CENTER, width=200)
    history_tree.heading(col, text=col)

# Add scrollbar for history
history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=history_tree.yview)
history_tree.configure(yscrollcommand=history_scrollbar.set)

# Pack history components
history_tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
history_scrollbar.pack(side="right", fill="y", pady=10)

# Initialize history table
update_history_table()

# Add price range info
price_info = f"Available price range: ₹{min_price:,.2f} - ₹{max_price:,.2f}"
price_info_label = tk.Label(input_frame, text=price_info, font=("Arial", 9))
price_info_label.grid(row=4, column=0, columnspan=2, pady=5)

root.mainloop()