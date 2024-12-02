import streamlit as st
import google.generativeai as genai
import PyPDF2
import sqlite3
import pandas as pd
import os
import time
from typing import List, Dict

class AdvancedPDFAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize Gemini client for advanced PDF analysis
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_content_with_retry(self, prompt: str, pdf_content: str, max_retries: int = 3) -> str:
        """
        Generate content with retry mechanism and custom prompting
        """
        # Truncate PDF content to manage token limits
        truncated_content = pdf_content[:50000]  # Limit to ~12,500 tokens

        # Combine user prompt with PDF content
        full_prompt = f"""
        PDF Document Context:
        {truncated_content}

        User's Specific Request:
        {prompt}

        Please provide a comprehensive and precise response based on the document and the user's request.
        """

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error processing request: {str(e)}"
                time.sleep(2 ** attempt)  # Exponential backoff
        return "Unable to process request"

class RealTimeDatabaseManager:
    def __init__(self, db_name='real_time_database.db', api_key: str = None):
        """
        Initialize database and PDF analyzer
        """
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Initialize PDF Analyzer
        self.pdf_analyzer = AdvancedPDFAnalyzer(api_key)
        
        # Create tables for PDF data storage
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                page_number INTEGER,
                content TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def extract_pdf_data(self, uploaded_file):
        """
        Extract text data from uploaded PDF
        """
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        extracted_data = []

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            full_text += page_text + "\n\n"
            
            # Insert each page's content into database
            self.cursor.execute('''
                INSERT INTO pdf_data 
                (filename, page_number, content) 
                VALUES (?, ?, ?)
            ''', (uploaded_file.name, page_num + 1, page_text))
            extracted_data.append({
                'filename': uploaded_file.name,
                'page_number': page_num + 1,
                'content': page_text
            })
        
        self.conn.commit()
        return extracted_data, full_text

    def get_pdf_content(self):
        """
        Retrieve all PDF content from the database
        """
        self.cursor.execute('SELECT content FROM pdf_data')
        return ' '.join([row[0] for row in self.cursor.fetchall()])

def get_google_api_key():
    """
    Retrieve Google API key with fallback methods
    """
    # Check environment variable first
    api_key = os.getenv('GOOGLE_API_KEY')
    
    # If no environment variable, prompt user directly in Streamlit
    if not api_key:
        api_key = st.sidebar.text_input(
            "Enter Google Gemini Pro API Key", 
            type="password", 
            help="Get your API key from Google AI Studio"
        )
    
    # Return the API key if provided
    return api_key

def main():
    st.set_page_config(page_title="PDF Insight Explorer", page_icon="📄", layout="wide")
    
    # Title and Introduction
    st.title("🔍 Advanced PDF Insight Explorer")
    st.markdown("""
    ### Powerful Data Analysis with Customizable AI Insights
    - Upload PDFs and extract powerful insights
    - Use predefined or custom AI analysis techniques
    - Get precise, context-aware responses
    """)

    # API Key Configuration
    api_key = get_google_api_key()
    if not api_key:
        st.warning("Please provide a valid Google Gemini Pro API Key")
        st.stop()

    # Initialize database manager
    db_manager = RealTimeDatabaseManager(api_key=api_key)

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # PDF Upload Section
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            with st.spinner('Processing PDF...'):
                try:
                    # Extract PDF data
                    extracted_data, full_text = db_manager.extract_pdf_data(uploaded_file)
                    st.success(f"Successfully uploaded {uploaded_file.name}")
                    st.write(f"Extracted {len(extracted_data)} pages")

                    # Custom Prompt Section
                    st.header("🤖 Custom AI Analysis")
                    custom_prompt = st.text_area(
                        "Enter your custom analysis request", 
                        placeholder="e.g., Extract all statistical data from the document",
                        height=150
                    )

                    # Predefined Analysis Options
                    analysis_options = [
                        "Summarize", 
                        "Document Information", 
                        "Step-by-Step Explanation", 
                        "Key Insights",
                        "Custom Analysis"
                    ]
                    analysis_type = st.selectbox("Choose Analysis Type", analysis_options)

                    if st.button("Analyze PDF"):
                        with st.spinner('Generating Advanced Analysis...'):
                            # Determine analysis approach
                            if analysis_type == "Custom Analysis" and not custom_prompt:
                                st.warning("Please enter a custom prompt for analysis")
                            else:
                                # Choose prompt based on selection
                                if analysis_type == "Custom Analysis":
                                    result = db_manager.pdf_analyzer.generate_content_with_retry(
                                        custom_prompt, 
                                        full_text
                                    )
                                else:
                                    # Use predefined analysis types
                                    result = db_manager.pdf_analyzer.analyze_pdf_content(
                                        full_text, 
                                        analysis_type.lower().replace(" ", "_")
                                    )
                                
                                # Display result with appropriate formatting
                                st.subheader(f"🔬 {analysis_type} Results")
                                st.markdown(result)

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    with col2:
        # Sidebar Information
        st.sidebar.header("📘 About Data Insight Explorer")
        st.sidebar.info("""
        **Advanced Features:**
        - Multi-page PDF processing
        - AI-powered document analysis
        - Custom prompt flexibility
        - Predefined analysis types
        - Secure API key management
        
        **Analysis Capabilities:**
        - Summarization
        - Document Information
        - Step-by-Step Breakdown
        - Key Insights Extraction
        - Fully Customizable Analysis
        """)

        # Example Prompts Section
        st.sidebar.header("💡 Example Prompts")
        example_prompts = [
            "Extract all numerical data",
            "Identify key arguments in the text",
            "Summarize the main conclusions",
            "List all technical terms",
            "Compare and contrast different sections"
        ]
        
        for prompt in example_prompts:
            st.sidebar.markdown(f"- {prompt}")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 Data Insight Explorer")

if __name__ == "__main__":
    main()
