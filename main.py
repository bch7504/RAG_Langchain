import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv  # Đọc file .env chứa API key
from seed_data import seed_milvus, seed_milvus_live  # Hàm xử lý dữ liệu
from agent import get_retriever, get_llm_and_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

def setup_sidebar():
    """
    Hàm tạo thanh công cụ tùy chọn
    """
    with st.sidebar:
        st.title(" Cấu hình")

        #Phần 1: Chọn embedding model
        st.header("Embedding model")
        embeddings_choice = st.radio(
            "Chọn embedding model",
            ["Ollama","HuggingFace"]
        )
        use_ollama = embeddings_choice == "Ollama"
        #Phần 2: Chọn nguồn dữ liệu
        st.header("Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File local","URL trực tiếp"]
        )
        if data_source == "File local":
            handle_local_file(use_ollama)
        else:
            handle_url_input(use_ollama)
       
        # Thêm phần chọn collection để query
        st.header("🔍 Collection để truy vấn")
        collection_to_query = st.text_input(
            "Nhập tên collection cần truy vấn:",
            "data_test",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )
       
        #Phần 3: Cấu hình LLM
        st.header("LLM model")
        llm_choice = st.radio(
            "Chọn LLM model",
            ["gemini","Ollama"]
        )
        
        return llm_choice, collection_to_query

def handle_local_file(use_ollama_embeddings: bool):
    """
    Xử lý khi người dùng chọn tải file
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")
    
    if st.button("Tải dữ liệu từ file"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang tải dữ liệu..."):
            try:
                seed_milvus(
                    'http://localhost:19530', 
                    collection_name, 
                    filename, 
                    directory, 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")
def handle_url_input(use_ollama_embeddings: bool):
    """
    Xử lý khi người dùng chọn crawl URL
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test_live",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    url = st.text_input("Nhập URL:", "https://www.stack-ai.com/docs")
    
    if st.button("Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang crawl dữ liệu..."):
            try:
                seed_milvus_live(
                    url, 
                    'http://localhost:19530', 
                    collection_name, 
                    'stack-ai', 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi crawl dữ liệu: {str(e)}")

def setup_chat_interface(llm_choice):
    st.title("💬 AI Assistant")
    
    # Caption động theo model
    if llm_choice == "Gemini":
        st.caption(" Trợ lý AI được hỗ trợ bởi LangChain và Gemini")
    else:
        st.caption(" Trợ lý AI được hỗ trợ bởi LangChain và Ollama LLaMA2")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs
def handle_user_input(msgs,agent_executor):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Stack AI!"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Gọi AI xử lý
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}
            )

            # Lưu và hiển thị câu trả lời
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)

def setup_page():
    """
    Hàm cấu hình web cơ bản
    """
    st.set_page_config(
        page_title="RAG LangChain",  # tiêu đề 
        page_icon="",
        layout="wide"  # giao diện rộng
    )

def initialize_app(): 
    """
    Hàm khởi tạo cấu hình trang và đọc .env chứa api
    """
    load_dotenv()
    setup_page()
    
def main():
    """
    Hàm chính điều khiển luồng chương trình
    """
    initialize_app()
    llm_choice, collection_to_query = setup_sidebar()
    msgs = setup_chat_interface(llm_choice)
    
    # Khởi tạo AI dựa trên lựa chọn model để trả lời
    if llm_choice == "Gemini":
        retriever = get_retriever(collection_to_query)
        agent_executor = get_llm_and_agent(retriever, "gemini")
    else:
        retriever = get_retriever(collection_to_query)
        agent_executor = get_llm_and_agent(retriever)

    handle_user_input(msgs, agent_executor)

    
if __name__ == "__main__":
    main()