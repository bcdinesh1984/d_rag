from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.language_models.chat_models import LangSmithParams
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails import LLMRails, RailsConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
from datetime import datetime

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

os.environ['LANGCHAIN_PROJECT'] = 'dinesh-end-to-end-rag'

class RAGAI:

    def __init__(self, name):
        self.name = name
        self.pdf_path = "C:/Users/bcdin/OneDrive/Documents/GenAI/submission.pdf"
        self.chunk_size = 1000 
        self.index_name = "pdf-index-name"
        self.search_service_name = "ragddsearch"
        self.as_api_key = os.getenv("AZURESEARCH_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.ls_api_key = os.getenv("LANGCHAIN_API_KEY")

        #print(self.openai_api_key)

        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key, model="text-embedding-3-small"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=100
        )

      

        self.loader = PyPDFLoader(self.pdf_path)

        self.endpoint = f"https://{self.search_service_name}.search.windows.net"
        self.vectorstore = SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=AzureKeyCredential(self.as_api_key))
        self.retriever = AzureAISearchRetriever(api_key=self.as_api_key, top_k=6, index_name=self.index_name,service_name=self.search_service_name)

        self.rag_prompt = hub.pull(
            "rlm/rag-prompt", 
            api_key=self.ls_api_key
        )

        self.gpt_llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            api_key=self.openai_api_key)
            
        

        self.config = RailsConfig.from_path("C:/Users/bcdin/OneDrive/Documents/GenAI/genaiprojects/rag/.venv/config")

        self.guardrails = RunnableRails(config=self.config,llm=self.gpt_llm)

               

    

    def generate_embeddings(self, text_chunks,query):
        print(' i am here')
        embeddings = self.embeddings
        i = int(datetime.now().timestamp())
       # print(text_chunks.page_content)
        for chunk in text_chunks:
            
            embedded_chunks = embeddings.embed_query(chunk)        
            documents = {"id": str(i), "content": chunk, "embeddings": embedded_chunks}
            
            self.vectorstore.upload_documents(documents=documents)
                       
            i += 1
        self.create_retrieval_chain(query)
        
    
    def extract_and_split_pdf(self):           
        pages = self.loader.load_and_split()
        chunks = []
        chunk_size=2000
        for page in pages:    
            
            text = page.page_content
            while len(text) > chunk_size:
                chunk, text = text[:chunk_size], text[chunk_size:]
                chunks.append(chunk)
            if text:              
              chunks.append(text)
        print(len(chunks))
        return chunks
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_retrieval_chain(self,query):        
        self.rag_chain = (                                            
                        {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}                               
                        | self.rag_prompt
                        | self.gpt_llm
                        | StrOutputParser()                        
                    )
        self.rag_chain =  self.guardrails | self.rag_chain  
        

    def qa(self, query,status):
        if status:
            chunks = self.extract_and_split_pdf()
            self.generate_embeddings(chunks,query)
        else:
            self.create_retrieval_chain(query)        
        return self.rag_chain.invoke(query),True
        


# if __name__ == "__main__":
#     rag = RAGAI("hi")
#     input = input("\n Enter the Question below :")    
#     a = rag.qa(input,False)
#     print(a)

    
  
