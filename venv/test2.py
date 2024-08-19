from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import AzureAISearchRetriever

load_dotenv(find_dotenv())

index_name = "pdf-index-name"
search_service_name = "ragddsearch"
as_api_key = os.getenv("AZURESEARCH_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
ls_api_key = os.getenv("LANGCHAIN_API_KEY")

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
# os.environ['LANGCHAIN_PROJECT'] = 'end-to-end-rag'

# Initialize the Langsmith client
llm = ChatOpenAI()

retriever = AzureAISearchRetriever(api_key=as_api_key, top_k=6, index_name=index_name,service_name=search_service_name)

rag_prompt = hub.pull(
            "rlm/rag-prompt", 
            api_key=os.getenv("LANGSMITH_API_KEY")
        )



config = RailsConfig.from_path("C:/Users/bcdin/OneDrive/Documents/GenAI/genaiprojects/rag/.venv/config")
guardrails = LLMRails(config=config,llm=llm)



def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
                        {
                           "context": retriever | format_docs,"question": RunnablePassthrough()
                        }
                        | rag_prompt
                        | llm
                        | StrOutputParser()
                    )
rg = rag_chain.invoke(input = "What is yashick's age")
guardrails.register_action(rg(), name='rg')
print(rg)