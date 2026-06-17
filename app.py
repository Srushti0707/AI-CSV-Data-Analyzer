import os
import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# Load environment variables from .env file
load_dotenv()

# --- CONSTANTS ---
DB_PATH = "history.db"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI CSV Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK & FEEL ---
st.markdown("""
    <style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek metric card container */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
    }
    
    /* Premium AI Insights styling */
    .ai-insight-card {
        background: linear-gradient(135deg, rgba(74, 119, 229, 0.12) 0%, rgba(100, 149, 237, 0.05) 100%);
        border-left: 6px solid #4a77e5;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
        color: var(--text-color);
        box-shadow: 0 4px 12px rgba(74, 119, 229, 0.05);
    }
    
    .ai-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        color: #4a77e5;
        font-size: 1.25rem;
        margin-bottom: 12px;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 8px;
    }
    
    /* Styled tags */
    .badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-success {
        background-color: rgba(40, 167, 69, 0.15);
        color: #28a745;
    }
    .badge-warning {
        background-color: rgba(220, 53, 69, 0.15);
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 1. DATABASE OPERATIONS
# ==========================================

def init_db(db_path=DB_PATH):
    """
    Initializes the SQLite database and creates the analysis_history table
    if it does not already exist.

    Args:
        db_path (str): Filepath to the SQLite database.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                summary TEXT NOT NULL
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database Initialization Error: {e}")
    finally:
        if conn:
            conn.close()


def save_session(filename, summary, db_path=DB_PATH):
    """
    Saves an analysis session's outcome to the SQLite history database.

    Args:
        filename (str): Name of the uploaded CSV file.
        summary (str): The AI-generated insights/summary text.
        db_path (str): Filepath to the SQLite database.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO analysis_history (filename, analysis_date, summary)
            VALUES (?, ?, ?)
        """, (filename, current_time, summary))
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database Save Error: {e}")
    finally:
        if conn:
            conn.close()


def get_history(db_path=DB_PATH):
    """
    Retrieves all stored analysis sessions sorted by date in descending order.

    Args:
        db_path (str): Filepath to the SQLite database.

    Returns:
        list of dicts: List containing previous analysis session details.
    """
    history = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        # Configure row factory to return dict-like objects
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, analysis_date, summary FROM analysis_history ORDER BY analysis_date DESC")
        rows = cursor.fetchall()
        for row in rows:
            history.append({
                "id": row["id"],
                "filename": row["filename"],
                "date": row["analysis_date"],
                "summary": row["summary"]
            })
    except sqlite3.Error as e:
        st.error(f"Database Query Error: {e}")
    finally:
        if conn:
            conn.close()
    return history


# ==========================================
# 2. DATA PROCESSING & CLEANING
# ==========================================

def load_data(uploaded_file):
    """
    Loads an uploaded CSV file safely into a Pandas DataFrame.

    Args:
        uploaded_file (UploadedFile): Streamlit file uploader buffer.

    Returns:
        pd.DataFrame: Loaded dataset, or None if reading failed.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Failed to read the CSV file. Details: {e}")
        return None


def clean_data(df, handling_method):
    """
    Cleans missing values in the dataset based on the user's operational choice.

    Args:
        df (pd.DataFrame): The original Pandas DataFrame.
        handling_method (str): Drop rows, fill with mean/mode, or preserve.

    Returns:
        pd.DataFrame: The cleaned Pandas DataFrame.
    """
    cleaned_df = df.copy()

    if handling_method == "Drop rows with missing values":
        cleaned_df = cleaned_df.dropna()
    elif handling_method == "Fill missing values (Mean/Mode)":
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # Numeric columns: Fill NaN with mean value
                mean_val = cleaned_df[col].mean()
                if not pd.isna(mean_val):
                    cleaned_df[col] = cleaned_df[col].fillna(mean_val)
            else:
                # Non-numeric columns: Fill NaN with mode value
                if not cleaned_df[col].mode().empty:
                    mode_val = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                else:
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")

    return cleaned_df


def get_numeric_stats(df):
    """
    Generates count, mean, minimum, and maximum statistics for numeric columns.

    Args:
        df (pd.DataFrame): Cleaned Pandas DataFrame.

    Returns:
        pd.DataFrame: Statistical details with metrics as columns and features as rows.
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        return pd.DataFrame()

    desc = numeric_df.describe()

    # Filter only the required statistics: count, mean, min, max
    stats_df = desc.loc[['count', 'mean', 'min', 'max']]

    # Transpose for easier horizontal scrolling and direct reading
    stats_df = stats_df.T
    stats_df = stats_df.rename(columns={
        'count': 'Count',
        'mean': 'Mean Value',
        'min': 'Minimum',
        'max': 'Maximum'
    })

    # Format counts as integer
    stats_df['Count'] = stats_df['Count'].astype(int)

    return stats_df


# ==========================================
# 3. GEMINI API INTEGRATION
# ==========================================

def get_gemini_insight(api_key, df, stats_df, filename):
    """
    Constructs data summary context and requests analysis insights from Gemini API.

    Args:
        api_key (str): Securely loaded Gemini API Key.
        df (pd.DataFrame): Cleaned Pandas DataFrame.
        stats_df (pd.DataFrame): Descriptive stats DataFrame.
        filename (str): Name of the CSV file.

    Returns:
        str: AI-generated analysis insight in markdown format.
    """
    if not api_key:
        return "Gemini API key is not configured. Please check your local `.env` configuration."

    try:
        # Initialize the modern google-genai Client
        client = genai.Client(api_key=api_key)

        # 1. Dataset Dimensions
        rows, cols = df.shape

        # 2. Extract column schema and details on missing values
        column_summaries = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = int(df[col].isna().sum())
            column_summaries.append(f"- **{col}**: Data Type `{dtype}`, missing {missing} value(s)")
        column_meta_text = "\n".join(column_summaries)

        # 3. Format numeric stats
        if not stats_df.empty:
            stats_markdown = stats_df.to_string()
        else:
            stats_markdown = "No numeric columns present in this dataset."

        # 4. Grab small head sample of dataset
        sample_markdown = df.head(5).to_string()

        # Compose prompt
        prompt = f"""
You are an expert Data Analyst AI. Review the following details of an uploaded CSV dataset named "{filename}" and provide clear, professional, plain-English insights.

### Dataset Overview:
- **Total Rows**: {rows}
- **Total Columns**: {cols}

### Column Structure:
{column_meta_text}

### Statistical Properties (Numeric Columns):
{stats_markdown}

### Sample Rows (Preview):
{sample_markdown}

### Analysis Requirements:
1. **High-Level Summary**: Describe in 2-3 sentences what this dataset appears to represent.
2. **Key Takeaways & Trends**: Note 3-4 key patterns, distributions, correlations, or anomalies visible in the statistics or sample data.
3. **Exploratory Directions**: Suggest 3 specific analytics questions that could be explored further with this data.

Write the output in clean, structured, beginner-friendly Markdown format with headings and bullet points. Avoid mentioning internal prompt details or any technical execution parameters. Output ONLY the analysis.
"""

        # API Call using gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

    except errors.APIError as e:
        return f"Gemini API Error: {e.message}"
    except Exception as e:
        return f"An unexpected error occurred during AI generation: {e}"


# ==========================================
# 4. STREAMLIT APPLICATION CONTROLLER
# ==========================================

def main():
    # Ensure database is initialized
    init_db()

    # Retrieve Gemini API Key from environment
    gemini_key = os.getenv("GEMINI_API_KEY")

    # Define state keys to prevent reset on button click triggers
    if "view_session" not in st.session_state:
        st.session_state.view_session = None
    if "current_insight" not in st.session_state:
        st.session_state.current_insight = None
    if "active_df" not in st.session_state:
        st.session_state.active_df = None
    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None
    if "active_filename" not in st.session_state:
        st.session_state.active_filename = None

    # --- SIDEBAR: HEADER AND KEY STATUS ---
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>📊 CSV Analyzer Controls</div>", unsafe_allow_html=True)

        # Secure indicator showing API configuration status without printing the key
        if gemini_key:
            st.markdown("Gemini API Status: <span class='badge badge-success'>Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("Gemini API Status: <span class='badge badge-warning'>Missing Key</span>", unsafe_allow_html=True)
            st.info("To use AI insights, copy `.env.example` to `.env` and configure your `GEMINI_API_KEY`.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Navigation Button to clear history view and focus back on active workspace
        if st.button("➕ Analyze New File", use_container_width=True):
            st.session_state.view_session = None
            st.session_state.current_insight = None
            st.session_state.active_df = None
            st.session_state.cleaned_df = None
            st.session_state.active_filename = None
            st.rerun()

        st.markdown("<br><div class='sidebar-header'>📅 Session History</div>", unsafe_allow_html=True)

        # Query database and render history list
        history_records = get_history()
        if not history_records:
            st.caption("No previous sessions stored.")
        else:
            for record in history_records:
                # Clickable history button
                label = f"📄 {record['filename']}\n({record['date'].split(' ')[0]})"
                if st.button(label, key=f"hist_{record['id']}", use_container_width=True):
                    st.session_state.view_session = record
                    st.rerun()

    # --- MAIN VIEW CONTROLLER ---

    # Scenario A: User is viewing a historical session from SQLite
    if st.session_state.view_session is not None:
        session = st.session_state.view_session
        st.title(f"Saved Session: {session['filename']}")
        st.caption(f"Analysis performed on: **{session['date']}**")

        # Back button to return to workspace
        if st.button("← Back to active workspace", type="secondary"):
            st.session_state.view_session = None
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Display the stored AI insights
        st.markdown("""
            <div class='ai-insight-card'>
                <div class='ai-header'>✨ Saved AI Analytical Insight</div>
                <div style='line-height: 1.6;'>
        """, unsafe_allow_html=True)
        st.markdown(session["summary"])
        st.markdown("</div></div>", unsafe_allow_html=True)

        return  # Terminate main layout execution for history page

    # Scenario B: Active workspace (upload and inspect new file)
    st.title("📊 AI-Powered CSV Data Analyzer")
    st.markdown("Upload any CSV file to instantly clean data, calculate statistics, and query Gemini API for analytical insights.")

    # Browser File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload any standard comma-separated file.")

    if uploaded_file is not None:
        # Load data on upload or if file changed
        if st.session_state.active_filename != uploaded_file.name:
            st.session_state.active_df = load_data(uploaded_file)
            st.session_state.active_filename = uploaded_file.name
            st.session_state.cleaned_df = None
            st.session_state.current_insight = None

        df = st.session_state.active_df

        if df is not None:
            # Layout columns for metadata and parameters
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("📋 Dataset Shape")
                st.markdown(f"**Rows**: {df.shape[0]} | **Columns**: {df.shape[1]}")
                missing_total = df.isna().sum().sum()
                st.markdown(f"**Total Missing Values**: {missing_total}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.subheader("🧹 Data Cleaning Parameters")
                clean_option = st.selectbox(
                    "Select missing value treatment:",
                    ["Keep missing values", "Drop rows with missing values", "Fill missing values (Mean/Mode)"]
                )

                # Apply data cleaning
                if st.session_state.cleaned_df is None:
                    st.session_state.cleaned_df = clean_data(df, clean_option)

                # Re-clean if option changes
                if st.button("Apply Cleaning & Recalculate"):
                    st.session_state.cleaned_df = clean_data(df, clean_option)
                    st.session_state.current_insight = None  # Clear old insights if data updates
                    st.toast("Cleaning applied successfully!", icon="✅")
                st.markdown("</div>", unsafe_allow_html=True)

            cleaned_df = st.session_state.cleaned_df

            # --- TABBED LAYOUT FOR PREVIEW AND STATS ---
            tab_preview, tab_stats = st.tabs(["📄 Data Preview", "🔢 Summary Statistics"])

            with tab_preview:
                st.subheader("Dataset Preview (Cleaned)")
                st.dataframe(cleaned_df, use_container_width=True)

            with tab_stats:
                st.subheader("Numeric Column Metrics")
                stats_df = get_numeric_stats(cleaned_df)

                if stats_df.empty:
                    st.warning("No numeric columns found in the dataset for statistics.")
                else:
                    st.dataframe(stats_df, use_container_width=True)

            # --- AI INSIGHT SECTION ---
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("✨ AI-Powered Analytics")

            if not gemini_key:
                st.warning("Please configure your `GEMINI_API_KEY` in the `.env` file to request AI analytical insights.")
            else:
                if st.button("🚀 Generate AI Insights", type="primary"):
                    with st.spinner("AI is analyzing dataset variables and computing patterns... Please wait..."):
                        numeric_summary = get_numeric_stats(cleaned_df)
                        insight = get_gemini_insight(gemini_key, cleaned_df, numeric_summary, uploaded_file.name)
                        st.session_state.current_insight = insight

                        # Store in history database
                        save_session(uploaded_file.name, insight)
                        st.toast("Analysis session stored successfully in database!", icon="💾")
                        st.rerun()  # Refresh layout to load new history element in sidebar

                # Render the current analysis insights if present
                if st.session_state.current_insight is not None:
                    st.markdown("""
                        <div class='ai-insight-card'>
                            <div class='ai-header'>✨ Gemini AI Insights & Recommendations</div>
                            <div style='line-height: 1.6;'>
                    """, unsafe_allow_html=True)
                    st.markdown(st.session_state.current_insight)
                    st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        # Initial Landing view when no file is uploaded
        st.info("👈 Upload a CSV file in the main view or select a previous analysis session from the sidebar to begin.")

        # Display short overview cards of what user can do
        st.markdown("<br><br><div class='sidebar-header'>🚀 What you can do</div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""
                <div class='metric-card'>
                    <h4>🧹 <b>Clean & Filter</b></h4>
                    <p>Handle missing records dynamically by either discarding incomplete rows or filling numeric/categorical values using mathematical measures.</p>
                </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown("""
                <div class='metric-card'>
                    <h4>🔢 <b>Summary Statistics</b></h4>
                    <p>Instantly view count, mean, minimum, and maximum values for every numeric column in your dataset.</p>
                </div>
            """, unsafe_allow_html=True)
        with col_c3:
            st.markdown("""
                <div class='metric-card'>
                    <h4>✨ <b>AI Data Analyst</b></h4>
                    <p>Send summary properties directly to Google Gemini and fetch natural language insights, trends, and exploratory guidelines instantly.</p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
