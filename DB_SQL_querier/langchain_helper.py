from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from few_shots import few_shots
import atexit

load_dotenv()

def extract_sql_query(llm_response: str) -> str:
    """
    Extract the SQL query from the LLM response, removing any additional text.
    
    Args:
        llm_response: Raw response from the LLM
        
    Returns:
        Clean SQL query string
    """
    # Common patterns in LLM responses
    patterns = [
        r"```sql\n(.*?)```",  # SQL between code blocks
        r"```(.*?)```",       # Any code blocks
        r"SELECT.*?;",        # SELECT statements
        r"INSERT.*?;",        # INSERT statements
        r"UPDATE.*?;",        # UPDATE statements
        r"DELETE.*?;"         # DELETE statements
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, llm_response, re.DOTALL | re.IGNORECASE)
        if matches:
            # Return the first match, stripped of whitespace
            return matches[0].strip()
    
    # If no patterns match, return the original stripped of whitespace
    return llm_response.strip()

def get_similar_examples(db, query, k=2):
    results = db.similarity_search(query, k=k)
    examples = []
    for doc in results:
        examples.append(f"""
Question: {doc.metadata['Question']}
SQLQuery: {doc.metadata['SQLQuery']}
SQLResult: {doc.metadata['SQLResult']}
Answer: {doc.metadata['Answer']}""")
    return "\n".join(examples)


custom_prompt = """You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run.
Unless otherwise specified, assume all regular text columns have a data type of VARCHAR(100).
Do not use any columns that do not exist in the table.

Here are some similar examples:
{examples}

Relevant tables:
{table_info}

Use {top_k} results.

Question: {{question}}"""

class DbConvo:
    def __init__(self):
        sql_db_user = "root"
        sql_db_password = os.getenv("DB_PASSWORD")
        sql_db_host = "localhost"
        sql_db_name = "atliq_tshirts"
        sql_db_connection_str = f"mysql+pymysql://{sql_db_user}:{sql_db_password}@{sql_db_host}/{sql_db_name}"
        self.sql_db = SQLDatabase.from_uri(sql_db_connection_str)
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.prompt = PromptTemplate(
            input_variables=["question", "table_info", "examples"],
            template=custom_prompt + PROMPT_SUFFIX
        )
        self.chain = create_sql_query_chain(
            self.llm,
            self.sql_db,
            prompt=self.prompt
        )
        self.chroma_db = self.create_chroma_db("new_db")
        
        # Register cleanup function
        atexit.register(self.cleanup)

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'llm') and hasattr(self.llm, '_client'):
                self.llm._client.close()
            if hasattr(self, 'sql_db') and hasattr(self.sql_db, 'engine'):
                self.sql_db.engine.dispose()
            # Remove persist() call as it's not needed
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()



    def create_chroma_db(self, name_of_dir):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        if name_of_dir not in os.listdir():
            few_shots_list = [" ".join(i.values()) for i in few_shots]
            document = []
            for i, shot in enumerate(few_shots_list):
                doc = Document(
                page_content=shot,
                metadata={
                    'Question': few_shots[i]['Question'],
                    'SQLQuery': few_shots[i]['SQLQuery'],
                    'SQLResult': few_shots[i]['SQLResult'],
                    'Answer': few_shots[i]['Answer']
                }   
                )
                document.append(doc)
            chroma_db = Chroma.from_documents(embedding=embeddings, documents=document, persist_directory=name_of_dir)
        else:
            chroma_db = Chroma(persist_directory=name_of_dir, embedding_function=embeddings)

        return chroma_db

    def get_answer(self, human_text):
        try:
            result_query = self.chain.invoke({
                "examples": get_similar_examples(self.chroma_db, human_text),
                "question": human_text,
                "table_info": self.sql_db.get_table_info()
            })
            query = extract_sql_query(result_query)
            print(f"Executing query: {query}")
            return self.sql_db.run(query)
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None









def main():
    """Main function to handle DB conversation"""
    db_convo = None
    try:
        db_convo = DbConvo()
        result = db_convo.get_answer("How many t-shirts do we have left?")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if db_convo is not None:
            db_convo.cleanup()


if __name__ == "__main__":
    main()