"""Defines the RAG chain for question answering."""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain

from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.core.config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_PROVIDER, HF_API_KEY
from src.core.vector_store_manager import Chroma # For retriever type hint

logger = logging.getLogger(__name__)


# Basic prompt template
QA_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}

Question: {question}

Helpful Answer:"""



def get_llm():
    """Initializes and returns a local Hugging Face LLM."""
    logger.info(f"Initializing local Hugging Face LLM with model: {LLM_MODEL_NAME}")

    if LLM_PROVIDER == "grok":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        return ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL_NAME)
    
    elif LLM_PROVIDER == "hugging_face":    
        if not HF_API_KEY:
            raise ValueError("HF_API_KEY not found in environment variables.")

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_auth_token=HF_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, use_auth_token=HF_API_KEY)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0,
            device="cpu",
            use_auth_token=HF_API_KEY)
        
        return HuggingFacePipeline(pipeline=pipe)


def create_rag_chain(retriever: Chroma.as_retriever):
    """Creates and returns the RAG chain."""
    logger.info("Creating RAG chain...")

    llm = get_llm()
    prompt = PromptTemplate.from_template(QA_TEMPLATE)

    # Summarization chain
    summarize_llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name='llama-3.1-8b-instant') # Use a fast model for summarization
    summary_chain = load_summarize_chain(llm=summarize_llm, chain_type="stuff") # Using "stuff" for simplicity with potentially many docs

    def summarize_docs(docs):
        # logger.info(f"Summarizing {len(docs)} documents...")
        # The summarize chain expects a list of Document objects and an input_documents key
        summary_result = summary_chain.invoke(docs) # Langchain summarizer can often handle list of docs directly
        # summary_result = summary_chain.invoke({"input_documents": docs}) # Alternative if direct list fails
        # logger.info(f"Summarized context: {summary_result['output_text']}")
        # Return as a single Document or string to fit into the existing format_docs logic
        # For now, let's return a single Document with the summarized content
        # This might need adjustment based on how format_docs expects its input
        # or format_docs might need to be updated to handle a single string summary.
        return [Document(page_content=summary_result['output_text'])]

    def format_docs(docs):
        # logger.info(f"Retrieved documents (post-summary): {docs}")
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | summarize_docs | format_docs, "question": RunnablePassthrough()} # type: ignore
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain created successfully with summarization.")
    return rag_chain


async def answer_question(chain, question: str) -> str:
    """Answers a question using the RAG chain."""
    logger.info(f"Answering question: {question} with {chain}")
    try:
        response = await chain.ainvoke(question)
        logger.info(f"Response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error invoking RAG chain: {e}", exc_info=True)
        return "Sorry, I encountered an error while processing your question."


# Example usage (for testing)
if __name__ == "__main__":
    import asyncio
    from src.core.vector_store_manager import (
        create_vector_store,
        get_retriever,
        load_vector_store,
    )
    from src.data_processing.data_loader import load_and_process_data
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv() # Ensure .env is loaded for config

    async def _test_rag_chain():
        logger.info("Testing RAG chain...")
        # 1. Load and process data
        split_articles, _ = await load_and_process_data()
        if not split_articles:
            logger.error("No articles loaded, cannot test RAG chain.")
            return

        # 2. Create or load vector store
        try:
            # vector_store = create_vector_store(split_articles) # Use if DB needs creation
            vector_store = load_vector_store() # Use if DB exists
        except Exception as e:
            logger.error(f"Failed to load/create vector store: {e}", exc_info=True)
            # As a fallback for testing, try creating if loading failed and it's a 'not found' type error
            logger.info("Attempting to create vector store as fallback...")
            try:
                vector_store = create_vector_store(split_articles)
            except Exception as e_create:
                logger.error(f"Fallback creation also failed: {e_create}", exc_info=True)
                return

        # 3. Get retriever
        retriever = await get_retriever(vector_store)

        # 4. Create RAG chain
        rag_chain = create_rag_chain(retriever) # type: ignore

        # 5. Test answering a question
        test_question = "Что такое цунами и каковы его основные причины?"
        answer = await answer_question(rag_chain, test_question)
        logger.info(f"Test Question: {test_question}")
        logger.info(f"Test Answer: {answer}")

        test_question_2 = "Кто был первым президентом США?"
        answer_2 = await answer_question(rag_chain, test_question_2)
        logger.info(f"Test Question 2: {test_question_2}")
        logger.info(f"Test Answer 2: {answer_2}")

    asyncio.run(_test_rag_chain())