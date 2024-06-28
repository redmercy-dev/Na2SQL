import streamlit as st
from sqlalchemy import (
    create_engine,
    inspect,
    select,
    text
)
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine
import openai
import os
import pandas as pd

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

class StreamlitChatPack:

    def __init__(
        self,
        page: str = "Natural Language to SQL Query",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize parameters."""
        self.page = page

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        st.set_page_config(
            page_title=f"{self.page}",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello. Ask me anything related to the database."}
            ]

        st.title(f"{self.page}💬")
        st.info("Hello to our AI powered SQL app. Pose any question and receive exact SQL queries.", icon="ℹ️")

        # Allow the user to upload a database file
        db_file = st.sidebar.file_uploader("Upload your Database", type=["db", "sql"])
        if db_file is not None:
            temp_db_path = "temp_uploaded_db.db"
            with open(temp_db_path, "wb") as f:
                f.write(db_file.getbuffer())

            try:
                # Attempt to create an engine assuming the file is a SQLite database
                engine = create_engine(f"sqlite:///{temp_db_path}")
                sql_database = SQLDatabase(engine)

                # Sidebar for database schema viewer
                st.sidebar.markdown("## Database Schema Viewer")
                inspector = inspect(engine)
                table_names = inspector.get_table_names()
                selected_table = st.sidebar.selectbox("Select a Table", table_names)

                # Function to get table data
                def get_table_data(table_name, conn):
                    query = f"SELECT * FROM {table_name}"
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
                llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

                # Initialize the Query Engine
                query_engine = NLSQLTableQueryEngine(
                    sql_database=sql_database, tables=[selected_table], llm=llm
                )

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

            except SQLAlchemyError as e:
                st.sidebar.error(f"Error processing the database: {str(e)}")
        else:
            st.sidebar.error("Please upload a database file.")

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
