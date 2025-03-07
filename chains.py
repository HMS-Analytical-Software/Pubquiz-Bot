from functools import partial

from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import format_document
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import AzureChatOpenAI


def create_map_reduce_chain(llm: AzureChatOpenAI, document_prompt: PromptTemplate, prompt: PromptTemplate) -> Runnable:
    partial_format_document = partial(format_document, prompt=document_prompt)

    # The chain we'll apply to each individual document.
    # Returns a summary of the document.
    map_chain = (
        {"context": partial_format_document}
        | prompt
        | llm
        | StrOutputParser()
    )

    # A wrapper chain to keep the original Document metadata
    map_as_doc_chain = (
        RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
        | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
    ).with_config(run_name="Summarize (return doc)")

    # The chain we'll repeatedly apply to collapse subsets of the documents
    # into a consolidate document until the total token size of our
    # documents is below some max size.
    def format_docs(docs):
        return "\n\n".join(partial_format_document(doc) for doc in docs)

    collapse_chain = (
        {"context": format_docs}
        | PromptTemplate.from_template("Collapse this content:\n\n{context}")
        | llm
        | StrOutputParser()
    )

    def get_num_tokens(docs):
        return llm.get_num_tokens(format_docs(docs))

    def collapse(
        docs,
        config,
        token_max=400,
    ):
        collapse_ct = 1
        while get_num_tokens(docs) > token_max:
            config["run_name"] = f"Collapse {collapse_ct}"
            invoke = partial(collapse_chain.invoke, config=config)
            split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
            docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
            collapse_ct += 1
        return docs

    # The chain we'll use to combine our individual document summaries
    # (or summaries over subset of documents if we had to collapse the map results)
    # into a final summary.

    reduce_chain = (
        {"context": format_docs}
        | PromptTemplate.from_template("Combine these summaries:\n\n{context}")
        | llm
        | StrOutputParser()
    ).with_config(run_name="Reduce")

    # The final full chain
    return (map_as_doc_chain.map() | collapse | reduce_chain).with_config(run_name="Map reduce")
