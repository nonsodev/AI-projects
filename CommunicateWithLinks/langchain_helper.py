from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import requests

load_dotenv()


class WebConvo:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_completion_tokens=200)
        self.prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Answer the question and cite the relevant sources from the context. if you do not know the answer, Say so!
    always return the cite if you cited in this format:
    f"Source: source_url'"
        f"Position: 'relative_position\"
        f"Content: a bit of page_content
        
        Simply say i do not know! if you dont know the answer, also do not cite anything if you don't know the answer""")

        self.embeddings = OpenAIEmbeddings()

    def load_and_chunk_web(self, urls: list):
        # Initialize empty list to store all documents
        all_documents = []

        # Load each URL separately to maintain source tracking
        for url in urls:
            try:
                loader = UnstructuredURLLoader(urls=[url])
                documents = loader.load()

                # Add URL to metadata for each document
                for doc in documents:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['source_url'] = url

                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading URL {url}: {str(e)}")
                continue

        # Create splitter with metadata handling
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            add_start_index=True,  # This adds chunk position information
        )

        # Split documents while preserving metadata
        chunks = splitter.split_documents(all_documents)

        # Add paragraph index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['paragraph_index'] = i
            # Calculate relative position in the document
            chunk.metadata['relative_position'] = f"Paragraph {i + 1} of {len(chunks)}"

        return chunks

    def create_retriever(self, chunks):
        # Create FAISS database with metadata
        faiss_db = FAISS.from_documents(chunks, self.embeddings)

        # Configure retriever to return metadata
        retriever = faiss_db.as_retriever(
            search_kwargs={
                "k": 3,
                "include_metadata": True  # Ensure metadata is included in search results
            }
        )

        return retriever

    def create_qa_chain(self, retriever):
        # First, format the context to include metadata
        def format_docs(docs):
            return "\n\n".join(
                f"Source: {doc.metadata['source_url']}\n"
                f"Position: {doc.metadata['relative_position']}\n"
                f"Content: {doc.page_content}"
                for doc in docs
            )

        chain = RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
        ) | self.prompt | self.llm | StrOutputParser()

        return chain
    
    def get_answer(self, query: str, urls: list[str]) -> str:
        """
        Get answer for a query using content from provided URLs.
        
        Args:
            query: The question to answer
            urls: List of URLs to source content from
            
        Returns:
            str: Answer to the query or error message
            
        Raises:
            ValueError: If query is empty or urls list is empty
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if not urls:
            raise ValueError("URLs list cannot be empty")

        try:
            # Load and chunk web content
            chunks = self.load_and_chunk_web(urls=urls)
            if not chunks:
                return "No content could be extracted from the provided URLs"

            # Create retriever and chain
            try:
                retriever = self.create_retriever(chunks)
            except Exception as e:
                return "Failed to process the content for retrieval"

            try:
                chain = self.create_qa_chain(retriever)
            except Exception as e:
                return "Failed to create question answering system"

            # Get answer
            try:
                return chain.invoke(query)
            except Exception as e:
                return "Failed to generate answer for the query"

        except requests.exceptions.RequestException as e:
            return "Failed to access one or more URLs"
        
        except Exception as e:
            return "An unexpected error occurred while processing your query"



