import os
import streamlit as st

from langchain_core.messages import SystemMessage
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import init_chatbot_tools


from dotenv import load_dotenv
load_dotenv("../.env", override=True)


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
    api_version="2024-06-01",
)

embeddings = AzureOpenAIEmbeddings(
    openai_api_key=azure_key,
    openai_api_version="2024-06-01",
    azure_deployment=azure_embeddings,
    azure_endpoint=azure_endpoint,
    chunk_size=16,
)

llm_reason = AzureChatOpenAI(
    api_key=azure_key,
    api_version=azure_version,
    azure_deployment=azure_reasoning,
    model=azure_deployment,
    azure_endpoint=azure_endpoint,
)

agent_prompt = SystemMessage(
    content="""
You are participating in a pubquiz.
Answer the question as short as possible.
Split a question in several tasks and try to solve them one by one.

Questions will implicetly either be about Tech Innovators or the MusicDb.
If not specified otherwise, try those sources beforehand.
Try to look at different sources if the answer is not found.

Remember to use the current date tool for any questions with temporal context!

Cite the source in brackets [<source>].

Before the final answer double check if you cited the sources!
""")

agent = create_react_agent(
    tools=init_chatbot_tools(llm, embeddings, llm_reason), model=llm, prompt=agent_prompt, debug=True,
)


def tool_call_from_message(message):
    if 'tool_calls' in message.additional_kwargs:
        call = message.additional_kwargs['tool_calls']
        try:
            call = message.additional_kwargs['tool_calls'][0]
            print(call)
            print(call['function'])
            return f"Called {call['function']['name']} with input {call['function']['arguments']}"
        except Exception as e:
            print(e)
    if message.content:
        return message.content
    return ""


if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = agent.invoke({
        "messages": prompt,
    })
    print('\n'.join([f"{message.__class__.__name__}: {tool_call_from_message(message)}" for message in response["messages"]]))
    st.chat_message("ai").write(response["messages"][-1].content)
