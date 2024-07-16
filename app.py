import streamlit as st
from sqlalchemy import create_engine, inspect
from typing import Dict, Any
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine
import os
import pandas as pd
import tempfile

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

class StreamlitChatPack:

    def __init__(
        self,
        page: str = "Natural Language to SQL Query",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        self.page = page

    def run(self, *args: Any, **kwargs: Any) -> Any:
        st.set_page_config(
            page_title=f"{self.page}",
            layout="wide",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hello. Ask me anything related to the database."}
            ]

        st.title(f"{self.page}💬")
        st.info("Welcome to our AI-powered SQL app. Upload your PostgreSQL database dump and ask questions about it.", icon="ℹ️")

        # Upload a database dump file
        db_file = st.sidebar.file_uploader("Upload your PostgreSQL Database Dump", type="sql")
        if db_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sql') as temp_file:
                temp_file.write(db_file.getvalue())
                temp_db_path = temp_file.name

            try:
                # Create a temporary PostgreSQL database
                temp_db_name = "temp_db_" + os.path.basename(temp_db_path).split('.')[0]
                engine = create_engine(f"postgresql://postgres:postgres@localhost:5432/postgres")
                with engine.connect() as conn:
                    conn.execute(f"CREATE DATABASE {temp_db_name}")
                    conn.close()

                # Connect to the new database
                engine = create_engine(f"postgresql://postgres:postgres@localhost:5432/{temp_db_name}")
                
                # Import the dump into the new database
                os.system(f"psql -U postgres -d {temp_db_name} -f {temp_db_path}")

                sql_database = SQLDatabase(engine)

                # Sidebar for database schema viewer
                st.sidebar.markdown("## Database Schema Viewer")
                inspector = inspect(engine)
                table_names = inspector.get_table_names()
                selected_table = st.sidebar.selectbox("Select a Table", table_names)

                # Function to get table data
                def get_table_data(table_name, conn):
                    query = f"SELECT * FROM {table_name} LIMIT 100"
                    df = pd.read_sql_query(query, conn)
                    return df

                # Display the selected table
                if selected_table:
                    with engine.connect() as conn:
                        df = get_table_data(selected_table, conn)
                    st.sidebar.text(f"Data preview for table '{selected_table}' (first 100 rows):")
                    st.sidebar.dataframe(df)

                # Initialize LLM with your desired settings
                llm = OpenAI(temperature=0.7, model="gpt-4")

                # Initialize the Query Engine
                query_engine = NLSQLTableQueryEngine(
                    sql_database=sql_database, tables=table_names, llm=llm
                )

                if "query_engine" not in st.session_state:
                    st.session_state["query_engine"] = query_engine

                # Chat interface
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state["messages"]:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])

                prompt = st.chat_input("Enter your natural language query about the database")
                if prompt:
                    with st.chat_message("user"):
                        st.write(prompt)
                    st.session_state["messages"].append({"role": "user", "content": prompt})

                    if st.session_state["messages"][-1]["role"] != "assistant":
                        with st.spinner("Generating response..."):
                            with st.chat_message("assistant"):
                                response = st.session_state["query_engine"].query("User Question:" + prompt + ". ")
                                sql_query = f"```sql\n{response.metadata['sql_query']}\n```\n**Response:**\n{response.response}\n"
                                st.write(sql_query)
                                st.session_state["messages"].append({"role": "assistant", "content": sql_query})

                                # Execute the generated SQL query and display results
                                with engine.connect() as conn:
                                    result_df = pd.read_sql_query(response.metadata['sql_query'], conn)
                                st.subheader("Query Result:")
                                st.dataframe(result_df)

            except Exception as e:
                st.error(f"Error processing the database: {str(e)}")
            finally:
                # Clean up: remove the temporary file and database
                os.unlink(temp_db_path)
                engine = create_engine(f"postgresql://postgres:postgres@localhost:5432/postgres")
                with engine.connect() as conn:
                    conn.execute(f"DROP DATABASE IF EXISTS {temp_db_name}")
                    conn.close()
        else:
            st.sidebar.error("Please upload a PostgreSQL database dump file.")

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
