from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize with a file path
loader = PyPDFLoader(file_path="C:/Users/bcdin/OneDrive/Documents/GenAI/Immunization.pdf")

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=100
        )
# Load and split the PDF into chunks
pages = loader.load_and_split()
#split_docs = text_splitter.split_documents(pages) 
# Each chunk is returned as a separate Document
chunks = []
chunk_size=100
for page in pages:                 
            
            text = page.page_content
            while len(text) > chunk_size:
                chunk, text = text[:chunk_size], text[chunk_size:]
                chunks.append(chunk)
            if text:
              
              chunks.append(text)
        #print(chunks)
print(len(chunks))