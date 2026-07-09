from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from seed_data import connect_to_milvus
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:
    """
    Tạo 1 ensemble retriever kết hợp vectorsearch ( Milvus) và BM25
    Args:
        collection_name: str: Tên collection trong Milvus
    Returns:
        EnsembleRetriever: Retriever kết hợp với tỷ trọng:
        - 70% Milvus vector search (k=4 kết quả)
        - 30% BM25 text search (k=4 kết quả)    
    """
    
    # Kết nối với Milvus và tạo vector retriever
    try:

        vectorstore = connect_to_milvus("http://localhost:19530",collection_name) 
        milvus_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})
    
        # Tạo BM25 retriever từ toàn bộ documents (nếu collection có dữ liệu)
        documents = [Document(page_content=doc.page_content, metadata=doc.metadata)
                    for doc in vectorstore.similarity_search("", k=100)]
    
        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
    
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4
        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7,0.3]
        )
    
        # Dùng milvus_retriever là chính
        return ensemble_retriever
    except Exception as e:
        print(f"Có lỗi khi khởi tạo retriever: {e}")
        # Trả về retriever với document mặc định khi có lỗi 
        default_doc=[
            Document(
                page_content="Có lỗi xảy ra khi kết nối với database. Vui lòng thử lại sau",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)
    
def get_llm_and_agent(retriever,llm_choice="gemini"):
    """
    Khởi tạo Language Model và Agent cấu hình cụ thể
    Args: 
        retriever: Retriever để tìm kiếm thông tin
    Return:
        CompiledStateGraph: Agent graph được cấu hình với:
        - Model: Gemini-2.5-flash 
        - Temperature: 0
        - Custom system prompt
       
    Chú ý:
        - Yêu cầu GOOGLE_API_KEY đã được cấu hình
        - Agent được thiết lập với "ChatbotAI"
        - Sử dụng chat history để duy trì ngữ cảnh hội thoại
    """
    
    # Tạo công cụ tìm kiếm cho agent
    tool = create_retriever_tool(
        retriever,
        "find",
        "Search for information of Stack-AI"
    )
    if llm_choice == "gemini":   
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            streaming=True,
            google_api_key=GOOGLE_API_KEY,
    )
    else:
        llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0,
            streaming=True,
        )
    
    tools = [tool]
        
    # Thiết lập system prompt cho agent
    system_prompt = "You are an expert at AI. Your name is ChatbotAI."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tạo và trả về agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent

# Gọi hàm tạo retriever và agent
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)