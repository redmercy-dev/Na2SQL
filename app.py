import streamlit as st
from sqlalchemy import create_engine, inspect, text
from typing import Dict, Any
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine
import openai
import os
import pandas as pd

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

class StreamlitChatPack:

    def __init__(self, page: str = "Natural Language to SQL Query", run_from_main: bool = False, **kwargs: Any) -> None:
        """Initialize parameters."""
        self.page = page

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        st.set_page_config(
            page_title=f"{self.page}",
            layout="centered",
            initial_sidebar_state="expanded",
            menu_items=None,
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello. Ask me anything related to the database."}
            ]

        st.title(f"{self.page}💬")
        st.info("Hello to our AI-powered SQL app. Pose any question and receive exact SQL queries.", icon="ℹ️")

        # Connection details for PostgreSQL
        st.sidebar.markdown("## Database Connection")
        db_host = st.sidebar.text_input("Host", value="localhost")
        db_port = st.sidebar.text_input("Port", value="5432")
        db_name = st.sidebar.text_input("Database Name")
        db_user = st.sidebar.text_input("Username")
        db_password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Connect to Database"):
            try:
                # Create an SQLAlchemy engine using the provided details
                engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
                sql_database = SQLDatabase(engine)

                # Sidebar for database schema viewer
                st.sidebar.markdown("## Database Schema Viewer")
                inspector = inspect(engine)
                table_names = inspector.get_table_names()
                selected_table = st.sidebar.selectbox("Select a Table", table_names)

                # Function to get table data
                def get_table_data(table_name, conn):
                    query = text(f"SELECT * FROM {table_name}")
                    df = pd.read_sql_query(query, conn)
                    return df

                # Display the selected table
                if selected_table:
                    conn = engine.connect()
                    df = get_table_data(selected_table, conn)
                    st.sidebar.text(f"Data for table '{selected_table}':")
                    st.sidebar.dataframe(df)
                    conn.close()

                # Initialize LLM with your desired settings
                llm = OpenAI(temperature=1, model="gpt-4")

                # Initialize the Query Engine
                query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=[selected_table], llm=llm)

                if "query_engine" not in st.session_state:
                    st.session_state["query_engine"] = query_engine

                for message in st.session_state["messages"]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

                if prompt := st.chat_input("Enter your natural language query about the database"):
                    with st.chat_message("user"):
                        st.write(prompt)
                    st.session_state["messages"].append({"role": "user", "content": prompt})

                    if st.session_state["messages"][-1]["role"] != "assistant":
                        with st.spinner():
                            with st.chat_message("assistant"):
                                response = st.session_state["query_engine"].query("User Question:" + prompt + ". ")
                                sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                                response_container = st.empty()
                                response_container.write(sql_query)
                                st.session_state["messages"].append({"role": "assistant", "content": sql_query})
            except Exception as e:
                st.sidebar.error(f"Failed to connect to database: {e}")
        else:
            st.sidebar.info("Enter your PostgreSQL connection details to connect.")

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
