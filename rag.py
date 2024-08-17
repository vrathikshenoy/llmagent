from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()

# Initialize LLM
llm = Ollama(model="llama3.1:8b", request_timeout=30.0)

# PDF Parser
parser = LlamaParse(result_type="markdown")

# CSV Reader class
class CSVReader:
    def load_data(self, file_path, extra_info=None):
        df = pd.read_csv(file_path)
        return [Document(text=df.to_string(), extra_info={"filename": str(file_path), **extra_info})]

# Updated file extractor
file_extractor = {
    ".pdf": parser,
    ".csv": CSVReader()
}

# Load documents (PDFs and CSVs)
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Initialize embedding model
embed_model = resolve_embed_model("local:BAAI/bge-m3")

# Function to ensure metadata is serializable
def serialize_metadata(metadata):
    return {k: str(v) if isinstance(v, PosixPath) else v for k, v in metadata.items()}

# Create vector index with serializable metadata
vector_index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    metadata_transform=serialize_metadata
)

# Create query engine
query_engine = vector_index.as_query_engine(llm=llm)

# Define tools
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="document_qa",
            description="This tool provides information from both PDF and CSV documents. Use this for querying the content of the documents.",
        ),
    )
]

# Create agent
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# Main loop
while (prompt := input("Enter prompt (or 'q' to quit): ")) != "q":
    response = agent.query(prompt)
    print(response)
