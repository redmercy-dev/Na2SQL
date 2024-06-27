import streamlit as st
from sqlalchemy import create_engine, inspect
from typing import Dict, Any

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    SQLDatabase,
)
from llama_index.base_pack import BasePack
from llama_index.llms import OpenAI
import openai
import os
import pandas as pd

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

class StreamlitChatPack(BasePack):

    def __init__(
        self,
        page: str = "Natural Language to SQL Query",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize parameters."""
        self.page = page

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}

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

        # Upload a database file
        db_file = st.sidebar.file_uploader("Upload your SQLite Database", type="db")
        if db_file is not None:
            # Save the uploaded file to a temporary location
            temp_db_path = "temp_uploaded_db.db"
            with open(temp_db_path, "wb") as f:
                f.write(db_file.getbuffer())

            # Create an SQLAlchemy engine using the uploaded file
            engine = create_engine(f"sqlite:///{temp_db_path}")

            sql_database = SQLDatabase(engine)  # Include all tables from the uploaded database

            # Initialize LLM with your desired settings
            llm2 = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")
            service_context = ServiceContext.from_defaults(llm=llm2, embed_model="local")

            # Sidebar for database schema viewer
            st.sidebar.markdown("## Database Schema Viewer")
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            selected_table = st.sidebar.selectbox("Select a Table", table_names)

            # Function to add messages to chat history
            def add_to_message_history(role, content):
                message = {"role": role, "content": str(content)}
                st.session_state["messages"].append(message)

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

            if "query_engine" not in st.session_state:
                st.session_state["query_engine"] = NLSQLTableQueryEngine(
                    sql_database=sql_database,
                    synthesize_response=True,
                    service_context=service_context
                )

            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if prompt := st.chat_input("Enter your natural language query about the database"):
                with st.chat_message("user"):
                    st.write(prompt)
                add_to_message_history("user", prompt)

                if st.session_state["messages"][-1]["role"] != "assistant":
                    with st.spinner():
                        with st.chat_message("assistant"):
                            response = st.session_state["query_engine"].query("User Question:"+prompt+". ")
                            sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                            response_container = st.empty()
                            response_container.write(sql_query)
                            add_to_message_history("assistant", sql_query)
        else:
            st.sidebar.error("Please upload a SQLite database file.")
            return

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
