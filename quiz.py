import os
import streamlit as st

from langchain_core.messages import SystemMessage
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv(override=True)


st.title("Pubquiz-Bot")

azure_version = "2024-06-01"
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_reasoning = os.getenv("AZURE_OPENAI_REASONING")
azure_embeddings = os.getenv("AZURE_OPENAI_EMBEDDINGS")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_KEY")

llm = AzureChatOpenAI(
    temperature=0.1,
    top_p=1.0,
    azure_deployment=azure_deployment,
    api_key=azure_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_version,
)

embeddings = AzureOpenAIEmbeddings(
    api_key=azure_key,
    api_version=azure_version,
    azure_deployment=azure_embeddings,
    azure_endpoint=azure_endpoint,
)

llm_reason = AzureChatOpenAI(
    api_key=azure_key,
    api_version=azure_version,
    azure_deployment=azure_reasoning,
    model=azure_deployment,
    azure_endpoint=azure_endpoint,
)

db = Chroma(persist_directory="./PubDatabase/chroma", embedding_function=embeddings)

agent_prompt = SystemMessage(content="""
You are participating in a pubquiz.
Answer the question as short as possible.
Cite the source in brackets [<source>].""")

agent = create_react_agent(
    tools=[], model=llm, prompt=agent_prompt,
)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = agent.invoke({
        "messages": prompt,
    })
    print('\n'.join([f"{message.__class__.__name__}: {str(message)}" for message in response["messages"]]))
    st.chat_message("ai").write(response["messages"][-1].content)
