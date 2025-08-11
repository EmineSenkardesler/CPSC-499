import pandas as pd
import streamlit as st
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Insurance Data Explorer", layout="wide")

# Title and description
st.title("Insurance Data Explorer")
st.markdown("Filter Risk Management Agency Cause of Loss data by commodity and cause of loss to explore variables.")

# Load data
@st.cache_data
def load_data():
    try:
        # Specify dtypes for problematic columns
        dtypes = {
            'insurance_plan_code': str,
            'insurance_abbreviation': str,
            'stage_code': str,
            'cause_of_loss_code': str
        }
        df = pd.read_csv('/home/emine2/DATA_ALL/colsom_1989_2024.csv', dtype=dtypes)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please upload your CSV file.")
        return None

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    dtypes = {
        'insurance_plan_code': str,
        'insurance_abbreviation': str,
        'stage_code': str,
        'cause_of_loss_code': str
    }
    df = pd.read_csv(uploaded_file, dtype=dtypes)
else:
    df = load_data()

if df is not None:
    # Sidebar for filters
    st.sidebar.header("Filters")
    
    # Commodity Name filter
    commodities = ['All'] + sorted(df['commodity_name'].unique().tolist())
    selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)
    
    # Cause of Loss Description filter
    causes = ['All'] + sorted(df['cause_of_loss_description'].unique().tolist())
    selected_cause = st.sidebar.selectbox("Select Cause of Loss", causes)
    
    # Filter data based on selections
    filtered_df = df.copy()
    if selected_commodity != 'All':
        filtered_df = filtered_df[filtered_df['commodity_name'] == selected_commodity]
    if selected_cause != 'All':
        filtered_df = filtered_df[filtered_df['cause_of_loss_description'] == selected_cause]
    
    # Display filtered data
    st.subheader("Filtered Data")
    # Select relevant columns for display
    display_columns = ['commodity_name', 'cause_of_loss_description', 'year_of_loss', 
                      'county_name', 'indemnity_amount', 'net_planted_quantity', 'loss_ratio']
    st.dataframe(filtered_df[display_columns], height=300)
    
    # Summary statistics
    if not filtered_df.empty:
        st.subheader("Summary Statistics")
        total_indemnity = filtered_df['indemnity_amount'].sum()
        avg_indemnity = filtered_df['indemnity_amount'].mean()
        total_planted = filtered_df['net_planted_quantity'].sum()
        st.write(f"**Total Indemnity Amount**: ${total_indemnity:,.2f}")
        st.write(f"**Average Indemnity Amount**: ${avg_indemnity:,.2f}")
        st.write(f"**Total Net Planted Quantity**: {total_planted:,.2f}")
        st.write(f"**Number of Records**: {len(filtered_df)}")
    
        # Visualization: Bar chart of indemnity by year
        st.subheader("Indemnity Amount by Year")
        yearly_indemnity = filtered_df.groupby('year_of_loss')['indemnity_amount'].sum().reset_index()
        fig = px.bar(yearly_indemnity, x='year_of_loss', y='indemnity_amount', 
                     title="Total Indemnity Amount by Year",
                     labels={'indemnity_amount': 'Indemnity Amount ($)', 'year_of_loss': 'Year'}, 
                     template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data matches the selected filters.")
else:
    st.warning("Please upload a CSV file to proceed.")

# Instructions
st.markdown("""
### Instructions
1. **Upload Data**: Use the file uploader to upload your CSV file, or ensure the CSV is at `/home/emine2/DATA_ALL/colsom_1989_2024.csv`.
2. **Filter Data**: Use the sidebar to select Commodity and Cause of Loss.
3. **Explore**: View the filtered data, summary statistics, and visualizations.
4. **Data Requirements**: Ensure your CSV has the required columns (e.g., `commodity_name`, `cause_of_loss_description`, `indemnity_amount`, `year_of_loss`).
""")