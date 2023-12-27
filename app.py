import streamlit as st
from sqlalchemy import create_engine, inspect, text
from typing import Dict, Any

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    download_loader,
)
from llama_index.llama_pack.base import BaseLlamaPack
from llama_index.llms import OpenAI
import openai
import os
import pandas as pd

from llama_index.llms.palm import PaLM

from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import sqlite3

from llama_index import SQLDatabase, ServiceContext
from llama_index.indices.struct_store import NLSQLTableQueryEngine

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


class StreamlitChatPack(BaseLlamaPack):

    def __init__(self, page: str = "Natural Language to SQL Query", run_from_main: bool = False, **kwargs: Any) -> None:
        """Initialize parameters."""
        self.page = page

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        st.set_page_config(page_title=f"{self.page}", layout="centered", initial_sidebar_state="auto", menu_items=None)

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello. Ask me anything related to the database."}]

        st.title(f"{self.page}💬")
        st.info("Hello to our AI powered SQL app. Pose any question and receive exact SQL queries.", icon="ℹ️")

        # Upload a database file
        db_file = st.sidebar.file_uploader("Upload your SQLite Database", type="db")

        if db_file is not None:
            # Save the uploaded file to a temporary location
            temp_db_path = "temp_uploaded_db.db"
            with open(temp_db_path, "wb") as f:
                f.write(db_file.getbuffer())

            # Create an SQLAlchemy engine using the uploaded file
            engine = create_engine(f"sqlite:///{temp_db_path}")
        else:
            engine = create_engine("sqlite:///ecommerce_platform1.db")  # Fallback to default DB if no file is uploaded

        # Sidebar for database schema viewer
        st.sidebar.markdown("## Database Schema Viewer")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        selected_table = st.sidebar.selectbox("Select a Table", table_names)

        if selected_table:
            df = pd.read_sql_table(selected_table, engine)
            st.sidebar.text(f"Data for table '{selected_table}':")
            st.sidebar.dataframe(df)
    
        # Close the connection
        conn.close()
              
        if "query_engine" not in st.session_state:  # Initialize the query engine
            st.session_state["query_engine"] = NLSQLTableQueryEngine(
                sql_database=sql_database,
                synthesize_response=True,
                service_context=service_context
            )

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])


        if prompt := st.chat_input(
            "Enter your natural language query about the database"
        ):  # Prompt for user input and save to chat history
            with st.chat_message("user"):
                st.write(prompt)
            add_to_message_history("user", prompt)

        # If last message is not from assistant, generate a new response
        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.spinner():
                with st.chat_message("assistant"):
                    response = st.session_state["query_engine"].query("User Question:"+prompt+". ")
                    sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                    response_container = st.empty()
                    response_container.write(sql_query)
                    # st.write(response.response)
                    add_to_message_history("assistant", sql_query)

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
