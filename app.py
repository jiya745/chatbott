import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables from the .env file
try:
    load_dotenv(encoding='utf-8')
except UnicodeDecodeError:
    st.error("Error reading .env file. Make sure it's UTF-8 encoded.")
    raise

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# Check if API keys are loaded properly
if not openai_api_key:
    st.error(" OPENAI_API_KEY not found in the .env file.")
    raise ValueError("OPENAI_API_KEY not found in .env file")

if not langchain_api_key:
    st.error(" LANGCHAIN_API_KEY not found in the .env file.")
    raise ValueError("LANGCHAIN_API_KEY not found in .env file")

# Set environment variables for Langchain
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Optional for LangSmith

# Initialize prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Initialize Streamlit UI
st.title('LangChain Demo with OpenAI API')
input_text = st.text_input("Ask a question or search a topic:")

# Initialize LLM and chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
chain = LLMChain(prompt=prompt, llm=llm)

# Run the chain when input is provided
if input_text:
    with st.spinner("Thinking..."):
        result = chain.run({'question': input_text})
        st.write("Response:")
        st.write(result)
