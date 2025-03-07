from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_openai.chat_models import AzureChatOpenAI

from .chains import create_map_reduce_chain


def create_map_reduce_summary_tool(llm: AzureChatOpenAI) -> Tool:
    summary_chain = create_map_reduce_chain(
        llm=llm,
        document_prompt=ChatPromptTemplate.from_template("""Content: {page_content}
Source: {source}"""),
        prompt=ChatPromptTemplate.from_template("""Summarize this content: Text: {context}"""),
    )
    return Tool(
        name="map_reduce_summary_tool",
        func=summary_chain.invoke,
        description="Use this tool to do a summary of very long texts. Make sure you get the text to do a summary of first."
    )
