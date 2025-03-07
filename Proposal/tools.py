from datetime import datetime
import base64
from mimetypes import guess_type
from typing import Dict

from langchain.prompts import HumanMessagePromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.llms.base import LLM
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain.tools import tool, Tool
from langchain_community.agent_toolkits.load_tools import load_tools

from sql_agent import create_sql_agent


@tool
def current_datetime() -> str:  # Note: A tool always needs an input and returns an output
    """Get the current date and time

    Returns:
        str: The current date and time
    """
    return datetime.now().strftime('%A %d %B %Y, %I:%M%p')


def init_stuff_doc_chain(llm: LLM):
    document_prompt = ChatPromptTemplate.from_template("""Content: {page_content}
Source: {source}""")

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Always cite the source in brackets [<source>]!

Question: {input}""")

    return create_stuff_documents_chain(
        llm=llm,
        document_prompt=document_prompt,
        prompt=prompt,
    )


def init_math_tool(llm: LLM) -> Tool:
    tool = load_tools(["llm-math"], llm=llm)[0]
    tool.description = "Use this tool to compute math tasks. Disregard any units, and work with numbers and operators. Source: Calculator"
    return tool


def init_reason_tool(llm: LLM) -> Tool:
    return Tool(
        name="reason",
        func=llm.invoke,
        description="Use this tool to solve logical or mathematical tasks and to reason about a given context. Source: Reasoning Model",
    )


def init_retriever_tools(llm: LLM, embeddings: LLM):
    tools = []
    doc_chain = init_stuff_doc_chain(llm)
    db = Chroma(persist_directory="./chroma/internal", embedding_function=embeddings)
    retriever = db.as_retriever(k=10)
    retrieval_internal_chain = {"input": RunnablePassthrough()} | create_retrieval_chain(retriever, doc_chain)
    tools.append(
        Tool(
            name="retrieval_internal",
            func=retrieval_internal_chain.invoke,
            description="Retrieve information from internal documents, employee list and guidelines from Tech Innovators.",
        )
    )

    db = Chroma(persist_directory="./chroma/reports", embedding_function=embeddings)
    retriever = db.as_retriever(k=20)
    retrieval_reports_chain = {"input": RunnablePassthrough()} | create_retrieval_chain(retriever, doc_chain)
    tools.append(
        Tool(
            name="retrieval_reports",
            func=retrieval_reports_chain.invoke,
            description="Retrieve reports about Tech Innovators. These reports contain financial information as well as major changes. Each report is for a specific year (2016-2023). If you need access to several years, list them all explicitly! When calling this tool, always add annual report!",  # noqa
        )
    )

    tools.append(
        Tool(
            name="retrieval_invoices",
            func=retrieval_reports_chain.invoke,
            description="Retrieve information from invoices from Tech Innovators. When calling this tool, always add invoice! Also, ensure to include keywords!",
        )
    )

    db = Chroma(persist_directory="./chroma/reports", embedding_function=embeddings)
    retriever = db.as_retriever(k=20)
    retrieval_reports_chain = {"input": RunnablePassthrough()} | create_retrieval_chain(retriever, doc_chain)
    tools.append(
        Tool(
            name="retrieval_user_feedbacks",
            func=retrieval_reports_chain.invoke,
            description="Retrieve information from user feedback for Tech Innovators. Each report is for a specific year (2016-2023). If you need access to several years, list them all explicitly! When calling this tool, always add feedback!",  # noqa
        )
    )

    db = Chroma(persist_directory="./chroma/guides", embedding_function=embeddings)
    retriever = db.as_retriever(k=10)
    retrieval_reports_chain = {"input": RunnablePassthrough()} | create_retrieval_chain(retriever, doc_chain)
    tools.append(
        Tool(
            name="retrieval_guides",
            func=retrieval_reports_chain.invoke,
            description="Retrieve information from user guides. 2016 and 2023 are available.",
        )
    )

    return tools


def init_image_tool(llm):
    # Function to encode a local image into data URL
    def local_image_to_data_url(image_path):
        mime_type, _ = guess_type(image_path)
        # Default to png
        if mime_type is None:
            mime_type = 'image/png'

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    prompt_template = HumanMessagePromptTemplate.from_template(
        template=[
            {"type": "text", "text": "{query}"},
            {
                "type": "image_url",
                "image_url": "{encoded_image_url}",
            },
        ]
    )

    summarize_image_prompt = ChatPromptTemplate.from_messages([prompt_template])
    gpt4_image_chain = summarize_image_prompt | llm

    def run_query_on_image(args: Dict[str, str]):
        return gpt4_image_chain.invoke(
            {
                "encoded_image_url": local_image_to_data_url(args["image"]),
                "query": args["query"],
            }
        )

    return Tool(
        name="image",
        func=run_query_on_image,
        description=(
            "Use this tool for a query with an image as context. Make sure you get the image url first! Pass the arguments (query, image) as json. Available images are:\n"  # noqa
            "../PubImages/UserGuide.jpg - An image which depicts the UserGuide 2016\n"
            "../PubImages/Employees.jpg - A graph which shows the employee growth from 2016 to 2023\n"
            "../PubImages/TeamBudget.jpg - A graph which shows the team event expenses from 2016 to 2023\n"
            "Cite the source in brackets [<source>].\n\n"
            "Example call: {{'query': '<your query>', 'image': '../PubImages/UserGuide.jpg'}}"
        ),
    )


def init_sql_agent_tool(llm: LLM) -> Tool:
    agent = create_sql_agent(llm)

    def invoke(args):
        response = agent.invoke({"messages": args})
        return response["messages"][-1].content

    return Tool(
        name="sql_agent",
        func=invoke,
        description="This is an agent designed to get answers from a database. Only ask questions in natural language! Reformulate the query to get a table as the result! Example: 'Tracks of Beyoncee'. Source: SQL Agent. Cite the tables!",  # noqa
    )


def init_chatbot_tools(llm, embeddings, llm_reason):
    tools = load_tools(["ddg-search"], llm=llm)
    tools.append(current_datetime)
    tools.append(init_math_tool(llm))
    tools.append(init_reason_tool(llm_reason))
    tools.extend(init_retriever_tools(llm, embeddings))
    tools.append(init_image_tool(llm))
    tools.append(init_sql_agent_tool(llm))

    return tools
