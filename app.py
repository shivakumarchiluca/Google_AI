import streamlit as st

# Session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Login page
if not st.session_state.authenticated:
    st.image("DILYTICS_LOGO.jpg", width=150)
    st.title("Welcome to Vertex AI Procurement Assistant")
    st.write("Please login to interact with your Procurement data")
    
    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", type="password")
    
    if st.button("Login"):
        # Simple validation (replace with your actual authentication logic)
        if username == "VERTEX-AI" and password == "Dilytics@123":
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main app content
else:
    import os
    import requests
    import json
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import re
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    
    # Set up page config
    st.set_page_config(
        page_title="Vertex AI - Procurement Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Configuration
    PROJECT_ID = "neural-proton-471612-u9"
    AGENT_ID = "70ec5718-730e-4ad0-a4bc-352c3b73c510"
    LOCATION = "us-central1"
    
    # ✅ Load service account credentials securely from Streamlit Secrets
    service_account_info = dict(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    @st.cache_resource
    def get_access_token():
        """Get access token for Google Cloud API"""
        try:
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            st.error(f"Failed to get access token: {str(e)}")
            return None
    
    def send_message_to_agent(message, session_id="streamlit-session"):
        """Send message to Vertex AI conversational agent using REST API"""
        access_token = get_access_token()
        
        if not access_token:
            return "Failed to get access token"
        
        try:
            url = f"https://{LOCATION}-dialogflow.googleapis.com/v3/projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ID}/sessions/{session_id}:detectIntent"
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "queryInput": {
                    "text": {
                        "text": message
                    },
                    "languageCode": "en"
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                if "queryResult" in result and "responseMessages" in result["queryResult"]:
                    for msg in result["queryResult"]["responseMessages"]:
                        if "text" in msg and "text" in msg["text"]:
                            return msg["text"]["text"][0]
                
                return "Hello! How can I help you with Procurement Insights today?"
            else:
                return f"API Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"
    
    def extract_data_from_response(response):
        """
        Extract structured data from the AI response.
        This function attempts to parse tables, key-value pairs, and structured data from text.
        """
        try:
            if not response or response in ["No response from agent", "Failed to get access token"]:
                return pd.DataFrame()
            
            lines = response.split('\n')
            table_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('|') and line.count('|') > 2: # At least 2 columns
                    table_lines.append(line)
            
            if not table_lines:
                return pd.DataFrame()
            
            # Parse the table lines
            data = []
            for line in table_lines:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if parts:
                    data.append(parts)
            
            if not data:
                return pd.DataFrame()
            
            # Assume first row is headers
            headers = data[0]
            data_rows = data[1:]
            
            # Skip separator line if present
            if data_rows and all(p.replace('-', '') == '' for p in data_rows[0]):
                data_rows = data_rows[1:]
            
            # Handle mismatched row lengths
            max_len = max(len(headers), max(len(row) for row in data_rows) if data_rows else 0)
            headers += [f"Column_{i+1}" for i in range(len(headers), max_len)]
            for i in range(len(data_rows)):
                data_rows[i] += [None] * (max_len - len(data_rows[i]))
            
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Convert numeric columns (assuming first column is categorical)
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with all NaN in numeric columns
            df = df.dropna(subset=df.columns[1:], how='all')
            
            return df
            
        except Exception as e:
            st.error(f"Error extracting data from response: {str(e)}")
            return pd.DataFrame()

    def extract_sql_query(response):
        """Extracts a SQL query from the response string."""
        match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def suggest_chart_types(df):
        """
        Suggest appropriate chart types based on the data structure
        """
        if df.empty:
            return ['No Chart']
        chart_types = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if len(numeric_columns) > 0 and len(categorical_columns) > 0:
            chart_types.extend(['Bar Chart', 'Line Chart', 'Pie Chart', 'Scatter Plot'])
        elif len(numeric_columns) > 1:
            chart_types.extend(['Scatter Plot', 'Line Chart', 'Bar Chart'])
        elif len(numeric_columns) == 1:
            chart_types.extend(['Histogram', 'Box Plot', 'Bar Chart'])
        if not chart_types:
            chart_types = ['Bar Chart', 'Line Chart', 'Pie Chart']
        return chart_types
    
    def create_visualization_with_selection(df, x_axis, y_axis, chart_type, prompt):
        """
        Create visualization based on user-selected x_axis, y_axis, and chart type
        """
        if df.empty or chart_type == 'No Chart' or not y_axis or y_axis not in df.columns:
            return None
        try:
            if x_axis == 'Index':
                df_plot = df.copy()
                df_plot['Index'] = df_plot.index
                x_col = 'Index'
            else:
                df_plot = df.copy()
                x_col = x_axis
            
            y_col = y_axis
            
            # Dynamic title based on column names
            title = f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}"
            # Enhance with prompt keywords if needed
            if "requisitions" in prompt.lower():
                title = f"Number of Requisitions by {x_col.replace('_', ' ').title()}"
            elif "billed" in prompt.lower():
                title = f"Total Billed Amount by {x_col.replace('_', ' ').title()}"
            
            if chart_type == 'Bar Chart':
                fig = px.bar(df_plot, x=x_col, y=y_col,
                             title=title,
                             labels={x_col: x_col.replace("_", " ").title(),
                                     y_col: y_col.replace("_", " ").title()})
            
            elif chart_type == 'Line Chart':
                fig = px.line(df_plot, x=x_col, y=y_col,
                              title=title,
                              labels={x_col: x_col.replace("_", " ").title(),
                                      y_col: y_col.replace("_", " ").title()})
            
            elif chart_type == 'Pie Chart':
                if x_col and x_col != 'Index':
                    df_pie = df_plot.nlargest(10, y_col) if len(df_plot) > 10 else df_plot
                    fig = px.pie(df_pie, values=y_col, names=x_col,
                                 title=title)
                else:
                    return None
            
            elif chart_type == 'Scatter Plot':
                fig = px.scatter(df_plot, x=x_col, y=y_col,
                                 title=title,
                                 labels={x_col: x_col.replace("_", " ").title(),
                                         y_col: y_col.replace("_", " ").title()})
            
            elif chart_type == 'Histogram':
                fig = px.histogram(df_plot, x=y_col,
                                   title=title,
                                   labels={y_col: y_col.replace("_", " ").title()})
            
            elif chart_type == 'Box Plot':
                fig = px.box(df_plot, y=y_col,
                             title=title,
                             labels={y_col: y_col.replace("_", " ").title()})
            
            else:
                fig = px.bar(df_plot, x=x_col, y=y_col,
                             title=title,
                             labels={x_col: x_col.replace("_", " ").title(),
                                     y_col: y_col.replace("_", " ").title()})
            
            fig.update_layout(
                template="plotly_white",
                height=400,
                showlegend=True,
                xaxis_tickangle=-45 if len(df_plot) > 5 else 0
            )
            
            if y_col in df_plot.columns and df_plot[y_col].notnull().any() and df_plot[y_col].max() > 10000:
                fig.update_layout(yaxis_tickformat='.2s')
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return None
    # Title
    st.title("Procurement Analytics Personal Assistant")
    # Sidebar
    with st.sidebar:
        st.image("VERTEXAI-GOOGLE LOGO.png", width=250)
        st.markdown("**About**")
        st.write(
        "This application leverages **Google Vertex AI** to help business users explore data through natural language. Simply ask a question, and it delivers relevant insights with clear answers and dynamic visualizations.It makes data analysis faster, more accessible, and empowers teams to monitor KPIs, spot trends, and drive smarter decisions.")
    # Add DILYTICS logo at the bottom
        st.markdown("**Powered by:**")
        st.image("DILYTICS_LOGO.jpg", width=200)

    
    # Chat interface
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = pd.DataFrame()
    if "chart_history" not in st.session_state:
        st.session_state.chart_history = []
    if "data_history" not in st.session_state:
        st.session_state.data_history = []
    
    # Display chat messages with charts
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content_text"])
            
            # Display SQL query in a dropdown
            if message["role"] == "assistant" and message["sql_query"]:
                with st.expander("SQL Query"):
                    st.code(message["sql_query"], language="sql")
            
            # Display chart for assistant messages if data exists
            if message["role"] == "assistant" and i < len(st.session_state.data_history):
                chart_data = st.session_state.data_history[i]
                if chart_data is not None and not chart_data.empty:
                    numeric_columns = chart_data.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_columns = chart_data.select_dtypes(include=['object']).columns.tolist()
                    
                    if numeric_columns and categorical_columns:
                        st.write("Visualization:")
                        suggested_charts = suggest_chart_types(chart_data)
                        
                        # Chart controls for this specific message
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_options = categorical_columns if categorical_columns else ['Index']
                            x_axis = st.selectbox(
                                "X axis",
                                options=x_options,
                                key=f"hist_x_axis_{i}"
                            )
                        with col2:
                            y_axis = st.selectbox(
                                "Y axis",
                                options=numeric_columns if numeric_columns else ['No numeric data'],
                                key=f"hist_y_axis_{i}"
                            )
                        with col3:
                            chart_type = st.selectbox(
                                "Chart Type",
                                options=suggested_charts,
                                key=f"hist_chart_type_{i}"
                            )
                        
                        # Create and display chart for this message
                        if chart_type and isinstance(chart_type, str):
                            if y_axis != 'No numeric data' and x_axis and y_axis:
                                try:
                                    fig = create_visualization_with_selection(
                                        chart_data.copy(),
                                        x_axis,
                                        y_axis,
                                        chart_type,
                                        message["prompt"] # Use the original prompt
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating visualization: {e}")
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content_text": prompt, "sql_query": None, "prompt": prompt})
        st.session_state.data_history.append(None)   # No data for user messages
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("AI Agent is thinking..."):
                response = send_message_to_agent(prompt)
                
                # Extract SQL query and clean the response text
                sql_query = extract_sql_query(response)
                content_text = response
                if sql_query:
                    content_text = re.sub(r'```sql\n.*?\n```', '', response, flags=re.DOTALL).strip()
                
                st.markdown(content_text)
                
                # Display SQL query in a dropdown immediately
                if sql_query:
                    with st.expander("SQL Query"):
                        st.code(sql_query, language="sql")
                
                # Try to extract data and store it in session state
                extracted_data = extract_data_from_response(content_text)
                st.session_state.extracted_data = extracted_data
                
                # Add data to history
                st.session_state.data_history.append(extracted_data.copy() if not extracted_data.empty else None)
                
                # Display chart immediately if data is available
                if not extracted_data.empty:
                    numeric_columns = extracted_data.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_columns = extracted_data.select_dtypes(include=['object']).columns.tolist()
                    
                    if numeric_columns and categorical_columns:
                        st.write("Visualization:")
                        suggested_charts = suggest_chart_types(extracted_data)
                        
                        # Current message index for unique keys
                        current_index = len(st.session_state.messages)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_options = categorical_columns if categorical_columns else ['Index']
                            x_axis = st.selectbox(
                                "X axis",
                                options=x_options,
                                key=f"current_x_axis_{current_index}"
                            )
                        with col2:
                            y_axis = st.selectbox(
                                "Y axis",
                                options=numeric_columns if numeric_columns else ['No numeric data'],
                                key=f"current_y_axis_{current_index}"
                            )
                        with col3:
                            chart_type = st.selectbox(
                                "Chart Type",
                                options=suggested_charts,
                                key=f"current_chart_type_{current_index}"
                            )
                        
                        # Create and display current chart
                        if chart_type and isinstance(chart_type, str):
                            if y_axis != 'No numeric data' and x_axis and y_axis:
                                try:
                                    fig = create_visualization_with_selection(
                                        extracted_data.copy(),
                                        x_axis,
                                        y_axis,
                                        chart_type,
                                        prompt
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating visualization: {e}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content_text": content_text, "sql_query": sql_query, "prompt": prompt})

        st.rerun()
