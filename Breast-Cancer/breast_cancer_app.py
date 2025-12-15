import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for hospital theme with consistent colors
st.markdown("""
<style>
    /* Main background and app container */
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        background-color: #f8fafc;
    }
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #1e293b;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
    }
    
    p, li, span {
        color: #475569;
        line-height: 1.6;
    }
    
    /* Prediction result boxes */
    .prediction-box {
        padding: 24px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        font-weight: 600;
        font-size: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .benign {
        background-color: #dcfce7;
        color: #166534;
        border: 2px solid #22c55e;
    }
    .malignant {
        background-color: #fef2f2;
        color: #dc2626;
        border: 2px solid #ef4444;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .info-box h3, .info-box h4 {
        color: #1e293b;
        margin-top: 0;
    }
    .info-box p, .info-box li {
        color: #475569;
        margin-bottom: 12px;
    }
    .info-box .highlight {
        color: #2563eb;
        font-weight: 600;
    }
    .info-box .benign-text {
        color: #059669;
        font-weight: 600;
    }
    .info-box .malignant-text {
        color: #dc2626;
        font-weight: 600;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 16px;
        border: 2px dashed #2563eb;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.1);
    }
    
    /* Data table styling */
    .dataframe {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .metric-card h4 {
        color: #64748b;
        font-size: 14px;
        font-weight: 500;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h2 {
        color: #1e293b;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    .metric-malignant h2 {
        color: #dc2626;
    }
    .metric-benign h2 {
        color: #059669;
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background-color: #f0fdf4;
        border: 1px solid #22c55e;
        color: #166534;
    }
    .stError {
        background-color: #fef2f2;
        border: 1px solid #ef4444;
        color: #dc2626;
    }
    .stWarning {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        color: #d97706;
    }
    .stInfo {
        background-color: #eff6ff;
        border: 1px solid #3b82f6;
        color: #1d4ed8;
    }
    
    /* Placeholder styling */
    .placeholder-box {
        background-color: #ffffff;
        padding: 32px;
        border-radius: 12px;
        text-align: center;
        border: 2px dashed #cbd5e1;
        margin-bottom: 24px;
    }
    .placeholder-box h4 {
        color: #475569;
        margin: 0 0 12px 0;
    }
    .placeholder-box p {
        color: #64748b;
        margin: 0;
    }
    
    /* Format info box */
    .format-info {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        border: 1px solid #e2e8f0;
    }
    .format-info h4 {
        color: #1e293b;
        margin-top: 0;
        margin-bottom: 16px;
    }
    .format-info p {
        color: #475569;
        margin-bottom: 12px;
        font-size: 14px;
    }
    .format-info strong {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Footer styling */
    .footer-box {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    .footer-box p {
        color: #475569;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .footer-box p:last-child {
        margin: 0;
        font-weight: 500;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 32px 0;
        margin-bottom: 32px;
    }
    .header-container h1 {
        color: #1e293b;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .header-container p {
        color: #2563eb;
        font-size: 20px;
        font-weight: 500;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Define the correct file paths
MODEL_PATH = r'C:\Users\ASUS\Intern-Pe-Updated\Breast-Cancer\breast_cancer_model.pkl'
SCALER_PATH = r'C:\Users\ASUS\Intern-Pe-Updated\Breast-Cancer\breast_cancer_scaler.pkl'

# Alternative: Check multiple possible locations
POSSIBLE_MODEL_PATHS = [
    r'C:\Users\ASUS\Intern-Pe-Updated\Breast-Cancer\breast_cancer_model.pkl',
    'breast_cancer_model.pkl',  # Current directory
    os.path.join(os.getcwd(), 'breast_cancer_model.pkl'),
    os.path.join(os.path.dirname(__file__), 'breast_cancer_model.pkl')
]

POSSIBLE_SCALER_PATHS = [
    r'C:\Users\ASUS\Intern-Pe-Updated\Breast-Cancer\breast_cancer_scaler.pkl',
    'breast_cancer_scaler.pkl',  # Current directory
    os.path.join(os.getcwd(), 'breast_cancer_scaler.pkl'),
    os.path.join(os.path.dirname(__file__), 'breast_cancer_scaler.pkl')
]

# Load the model
@st.cache_resource
def load_model():
    """Load the trained model from the saved pickle file"""
    model = None
    model_path_used = None
    
    # Try different possible paths
    for path in POSSIBLE_MODEL_PATHS:
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                model_path_used = path
                st.success(f"✅ Model loaded successfully from: {path}")
                break
        except Exception as e:
            st.warning(f"⚠️ Could not load model from {path}: {str(e)}")
            continue
    
    if model is None:
        st.error("❌ Could not find the trained model file. Please ensure the model is saved correctly.")
        st.error("Expected locations:")
        for path in POSSIBLE_MODEL_PATHS:
            st.write(f"- {path}")
        st.stop()
    
    return model, model_path_used

# Load the scaler
@st.cache_resource
def load_scaler():
    """Load the trained scaler from the saved pickle file"""
    scaler = None
    scaler_path_used = None
    
    # Try different possible paths
    for path in POSSIBLE_SCALER_PATHS:
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    scaler = pickle.load(f)
                scaler_path_used = path
                st.success(f"✅ Scaler loaded successfully from: {path}")
                break
        except Exception as e:
            st.warning(f"⚠️ Could not load scaler from {path}: {str(e)}")
            continue
    
    if scaler is None:
        st.error("❌ Could not find the trained scaler file. Please ensure the scaler is saved correctly.")
        st.error("Expected locations:")
        for path in POSSIBLE_SCALER_PATHS:
            st.write(f"- {path}")
        st.stop()
    
    return scaler, scaler_path_used

# Function to make predictions
def predict_batch(data, model, scaler):
    """Make predictions on a batch of data"""
    try:
        # Validate input data shape
        if data.shape[1] != 30:
            st.error(f"❌ Input data has {data.shape[1]} features, but model expects 30 features")
            return None, None
        
        # Standardize the input features
        data_std = scaler.transform(data)
        
        # Make predictions
        predictions = model.predict(data_std, verbose=0)  # Set verbose=0 to reduce output
        
        # Get the class with highest probability for each sample
        predicted_classes = np.argmax(predictions, axis=1)
        probabilities = np.max(predictions, axis=1)
        
        return predicted_classes, probabilities
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None

# Expected column order (30 features)
EXPECTED_COLUMNS = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error', 'concave points error',
    'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture',
    'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
    'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def validate_csv_data(df):
    """Validate the uploaded CSV data"""
    issues = []
    
    # Check number of columns
    if len(df.columns) != 30:
        if len(df.columns) == 31:
            # Check if first column might be an ID column
            first_col = df.iloc[:, 0]
            if first_col.dtype == 'object' or all(isinstance(x, (int, str)) for x in first_col[:5]):
                issues.append("warning:Found 31 columns - assuming first column is ID and will be removed")
            else:
                issues.append(f"error:Expected 30 feature columns, found {len(df.columns)}")
        else:
            issues.append(f"error:Expected 30 feature columns, found {len(df.columns)}")
    
    # Check for missing values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        issues.append(f"error:Missing values found in columns: {list(null_cols.index)}")
    
    # Check data types
    non_numeric_cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        issues.append(f"error:Non-numeric columns found: {non_numeric_cols}")
    
    return issues

def main():
    # Load model and scaler
    try:
        model, model_path = load_model()
        scaler, scaler_path = load_scaler()
    except:
        st.error("❌ Failed to load model or scaler. Please check the file paths and ensure the files exist.")
        return
    
    # Header
    st.markdown("""
    <div class='header-container'>
        <h1>🏥 Breast Cancer Prediction System</h1>
        <p>Upload CSV file for batch prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show model information
    with st.expander("🔧 Model Information"):
        st.write(f"**Model Path:** {model_path}")
        st.write(f"**Scaler Path:** {scaler_path}")
        st.write(f"**Expected Features:** 30")
        st.write(f"**Model Input Shape:** (None, 30)")
    
    # Information box
    with st.expander("ℹ️ About this application"):
        st.markdown("""
        <div class='info-box'>
            <h3>About This Tool</h3>
            <p>This application uses <span class='highlight'>machine learning</span> to predict whether breast tumors are <span class='benign-text'>benign</span> or <span class='malignant-text'>malignant</span> based on features extracted from breast fine needle aspirate (FNA) images.</p>
            <p>The model has been trained on the Wisconsin Breast Cancer dataset and achieves over <span class='highlight'>93% accuracy</span> in predictions.</p>
            
            <h4>CSV File Requirements:</h4>
            <ul>
                <li>Must contain exactly 30 feature columns (numerical values only)</li>
                <li>No header row required (but acceptable if present)</li>
                <li>Each row represents one patient sample</li>
                <li>Features should be in the standard Wisconsin Breast Cancer dataset order</li>
            </ul>
            
            <div style='background-color:#eff6ff;padding:16px;border-radius:8px;margin-top:20px;border-left:4px solid #2563eb;'>
                <p style='margin:0;color:#1e293b;'><b>Note:</b> This tool is for educational purposes only and should not replace professional medical advice.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3>File Upload</h3>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing patient data with 30 features"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file, header=None)  # Read without header first
                
                # Display file info
                st.success(f"✅ File uploaded successfully!")
                st.info(f"📊 File contains {len(df)} rows and {len(df.columns)} columns")
                
                # Validate the data
                validation_issues = validate_csv_data(df)
                
                # Handle validation issues
                has_errors = False
                for issue in validation_issues:
                    if issue.startswith("error:"):
                        st.error(f"❌ {issue[6:]}")
                        has_errors = True
                    elif issue.startswith("warning:"):
                        st.warning(f"⚠️ {issue[8:]}")
                
                if has_errors:
                    st.stop()
                
                # Handle the data based on number of columns
                if len(df.columns) == 31:
                    st.info("🔧 Removing first column (assumed to be ID)")
                    df = df.iloc[:, 1:]  # Remove first column
                elif len(df.columns) == 30:
                    st.success("✅ Correct number of features (30)")
                
                # Assign column names
                df.columns = EXPECTED_COLUMNS
                
                # Display sample of the data
                st.markdown("<h4>Data Preview</h4>", unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)
                
                # Show data statistics
                st.markdown("<h4>Data Statistics</h4>", unsafe_allow_html=True)
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    st.metric("Total Samples", len(df))
                    st.metric("Total Features", len(df.columns))
                with col1_2:
                    st.metric("Missing Values", df.isnull().sum().sum())
                    st.metric("Data Type", "Numeric" if df.select_dtypes(include=[np.number]).shape[1] == 30 else "Mixed")
                
                # Prediction button
                if st.button("🔍 Predict All Samples", type="primary"):
                    with st.spinner('Making predictions...'):
                        # Make predictions
                        predicted_classes, probabilities = predict_batch(df.values, model, scaler)
                        
                        if predicted_classes is not None and probabilities is not None:
                            # Create results dataframe
                            results_df = df.copy()
                            results_df['Prediction'] = ['Malignant' if pred == 0 else 'Benign' for pred in predicted_classes]
                            results_df['Confidence'] = [f"{prob*100:.2f}%" for prob in probabilities]
                            results_df['Risk_Level'] = ['High' if pred == 0 else 'Low' for pred in predicted_classes]
                            
                            # Store results in session state
                            st.session_state.results_df = results_df
                            st.session_state.predicted_classes = predicted_classes
                            st.session_state.probabilities = probabilities
                            
                            st.success("✅ Predictions completed!")
                        else:
                            st.error("❌ Prediction failed. Please check your data and try again.")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please make sure your CSV file is properly formatted with numerical data only.")
    
    with col2:
        st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
        
        # Display results if available
        if 'results_df' in st.session_state:
            results_df = st.session_state.results_df
            predicted_classes = st.session_state.predicted_classes
            probabilities = st.session_state.probabilities
            
            # Summary statistics
            malignant_count = sum(predicted_classes == 0)
            benign_count = sum(predicted_classes == 1)
            total_samples = len(predicted_classes)
            
            # Display summary
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>Total Samples</h4>
                    <h2>{total_samples}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2_2:
                st.markdown(f"""
                <div class='metric-card metric-malignant'>
                    <h4>Malignant</h4>
                    <h2>{malignant_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2_3:
                st.markdown(f"""
                <div class='metric-card metric-benign'>
                    <h4>Benign</h4>
                    <h2>{benign_count}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Results table
            st.markdown("<h4>Detailed Results</h4>", unsafe_allow_html=True)
            
            # Display only the last few columns for cleaner view
            display_df = results_df[['Prediction', 'Confidence', 'Risk_Level']].copy()
            display_df.index = range(1, len(display_df) + 1)  # Start index from 1
            display_df.index.name = 'Sample'
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualization
            st.markdown("<h4>Prediction Distribution</h4>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            # Create pie chart
            labels = ['Benign', 'Malignant']
            sizes = [benign_count, malignant_count]
            colors = ['#059669', '#dc2626']
            explode = (0.05, 0.05)  # explode slices slightly
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            explode=explode, shadow=True, startangle=90,
                                            textprops={'fontsize': 14, 'color': '#1e293b', 'fontweight': '600'})
            
            ax.set_title('Prediction Distribution', fontsize=18, color='#1e293b', fontweight='bold', pad=20)
            
            # Make percentage text bold and white for visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(16)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download results
            st.markdown("<h4>Download Results</h4>", unsafe_allow_html=True)
            
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:
            # Show placeholder when no file is uploaded
            st.markdown("""
            <div class='placeholder-box'>
                <h4>📁 Upload CSV File</h4>
                <p>Upload a CSV file on the left to see prediction results here.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample file information
            st.markdown("""
            <div class='format-info'>
                <h4>📋 Expected CSV Format</h4>
                <p>Your CSV should have exactly 30 columns with the following features:</p>
                <p><strong>Mean features (10):</strong> radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension</p>
                <p><strong>Standard error features (10):</strong> Same as above but for standard errors</p>
                <p><strong>Worst features (10):</strong> Same as above but for worst/largest values</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer-box'>
        <p>This application is for educational purposes only and should not be used for medical diagnosis.</p>
        <p>Always consult with healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()