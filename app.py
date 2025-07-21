import streamlit as st
import pandas as pd
import joblib
import sklearn # Important for joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
# Use the full screen width
st.set_page_config(page_title="Car Price Dashboard", page_icon="📊", layout="wide")

# --- Data Loading ---
# Cache the data loading to improve performance
@st.cache_data
def load_data(path):
    """Loads the car price dataset."""
    df = pd.read_csv(path)
    return df

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    """Loads the trained machine learning model."""
    model = joblib.load(path)
    return model

# Load data and model
try:
    df_full = load_data('car-price_cleaned.csv')
    model = load_model('car_price_model.joblib')
except FileNotFoundError:
    st.error("A required file was not found. Please ensure 'car-price_cleaned.csv' and 'car_price_model.joblib' are in the same folder.")
    st.stop()

# --- App Structure ---
st.title("📊 Car Price Prediction Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["Prediction", "Data Exploration"])


# --- TAB 1: PREDICTION ---
with tab1:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Enter Car Features")

    def user_inputs():
        """Create the sidebar widgets and return the inputs in a DataFrame."""
        brands = sorted(df_full['Brand'].unique())
        models = sorted(df_full['Model'].unique())
        fuel_types = sorted(df_full['Fuel_Type'].unique())
        transmissions = sorted(df_full['Transmission'].unique())

        brand = st.sidebar.selectbox("Brand", brands)
        model_car = st.sidebar.selectbox("Model", models)
        year = st.sidebar.slider("Year", int(df_full['Year'].min()), int(df_full['Year'].max()), 2018)
        engine_size = st.sidebar.slider("Engine Size", float(df_full['Engine_Size'].min()), float(df_full['Engine_Size'].max()), 2.0, 0.1)
        fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types)
        transmission = st.sidebar.selectbox("Transmission", transmissions)
        mileage = st.sidebar.slider("Mileage (km)", 0, 300000, 100000, 1000)
        doors = st.sidebar.select_slider("Number of Doors", options=[2, 3, 4, 5])
        owner_count = st.sidebar.select_slider("Number of Previous Owners", options=[1, 2, 3, 4, 5])

        data = {
            'Brand': brand, 'Model': model_car, 'Year': year, 'Engine_Size': engine_size,
            'Fuel_Type': fuel_type, 'Transmission': transmission, 'Mileage': mileage,
            'Doors': doors, 'Owner_Count': owner_count
        }
        features_df = pd.DataFrame(data, index=[0])
        return features_df

    input_df = user_inputs()

    # --- Prediction Display Area ---
    st.header("Price Prediction")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Your Selections:")
        st.write(input_df)
        
        # Estimate button
        if st.button("Estimate Price", type="primary", use_container_width=True):
            prediction = model.predict(input_df)
            price_prediction = round(prediction[0], 2)

            # Store the prediction in session state to use it across the app
            st.session_state['price_prediction'] = price_prediction
        
    with col2:
        st.subheader("Prediction Result")
        # Display the result if it exists in the session state
        if 'price_prediction' in st.session_state:
            price = st.session_state['price_prediction']
            
            # Use a metric for a clean, professional look
            st.metric(label="Estimated Price", value=f"{price:,.0f} €")
            
            # Create a professional-looking gauge chart
            min_price = df_full['Price'].min()
            max_price = df_full['Price'].max()
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = price,
                title = {'text': "Price Gauge"},
                gauge = {
                    'axis': {'range': [min_price, max_price], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#0047AB"}, # Cobalt Blue
                    'steps' : [
                        {'range': [min_price, df_full['Price'].quantile(0.33)], 'color': "lightgray"},
                        {'range': [df_full['Price'].quantile(0.33), df_full['Price'].quantile(0.66)], 'color': "gray"}],
                }))
            st.plotly_chart(fig, use_container_width=True)

    st.info("Disclaimer: This is an estimated price based on a machine learning model and should be used as a reference only.", icon="ℹ️")

# --- TAB 2: DATA EXPLORATION ---
with tab2:
    st.header("Explore the Car Market")
    st.markdown("Here are some insights from the dataset used to train the model.")

    # Create two columns for charts
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        # Chart 1: Price Distribution
        st.subheader("Distribution of Car Prices")
        fig_hist = px.histogram(df_full, x="Price", nbins=50, title="Car Price Distribution", color_discrete_sequence=['#0047AB'])
        st.plotly_chart(fig_hist, use_container_width=True)

        # Chart 2: Average Price by Brand
        st.subheader("Average Price by Brand")
        avg_price_brand = df_full.groupby('Brand')['Price'].mean().sort_values(ascending=False).reset_index()
        fig_bar = px.bar(avg_price_brand, x='Brand', y='Price', title="Average Price per Brand", color='Brand')
        st.plotly_chart(fig_bar, use_container_width=True)

    with fig_col2:
        # Chart 3: Price vs. Mileage
        st.subheader("Price vs. Mileage")
        fig_scatter = px.scatter(df_full, x="Mileage", y="Price", color="Fuel_Type",
                                 title="Price vs. Mileage by Fuel Type",
                                 hover_data=['Brand', 'Model', 'Year'])
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Chart 4: Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        numeric_df = df_full.select_dtypes(include= 'number')
        corr = numeric_df.corr()
        fig_heatmap, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Correlation Between Features")
        st.pyplot(fig_heatmap)