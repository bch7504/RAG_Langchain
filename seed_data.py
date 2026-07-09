import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from dotenv import load_dotenv
from uuid import uuid4
from crawl import crawl_web
from langchain_ollama import OllamaEmbeddings
from pymilvus import MilvusClient, connections


load_dotenv()

def _ensure_orm_connection(vectorstore: Milvus):
    """
    Workaround cho bug langchain-milvus 0.3.3 + pymilvus 2.6.x:
    MilvusClient tạo connection nội bộ nhưng không đăng ký với pymilvus ORM,
    nên Collection() không tìm thấy alias và raise ConnectionNotExistException.
    Hàm này đăng ký ORM connection với alias mà MilvusClient đang dùng.
    """
    alias = vectorstore.alias
    # Kiểm tra xem alias đã được đăng ký trong ORM chưa
    if alias not in connections.list_connections() or not connections.has_connection(alias):
        # Lấy URI từ connection_args
        uri = vectorstore._connection_args.get("uri", "http://localhost:19530")
        connections.connect(alias=alias, uri=uri)

def _get_embeddings(use_ollama: bool):
    """Khởi tạo model embeddings dựa trên lựa chọn"""
    if use_ollama:
        return OllamaEmbeddings(
            model="nomic-embed-text"  
        )
    else: 
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

def _drop_collection_if_exists(uri: str, collection_name: str):
    """Xóa collection cũ nếu đã tồn tại (workaround bug drop_old langchain-milvus 0.3.x)"""
    client = MilvusClient(uri=uri)
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        print(f'Dropped old collection: {collection_name}')
    client.close()

def load_data_from_local_file(filename: str, directory: str) -> tuple:
    """
    Hàm đọc dữ liệu từ file JSON local
    Args:
        filename: str: Tên file JSON cần đọc (vd: data.json)
        directory: str: Thư mục chứa file ( vd: data_v3)
    Returns: 
        tuple: Trả về (data,doc_name) trong đó:
            - data: Dữ liệu JSON đã được parse
            - doc_name: Tên file đã được xử lý (bỏ đuôi .json và thay '_' bằng khoảng trống)
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    # Chuyển tên file thành tên tài liệu (bỏ đuôi .json và thay '_' bằng khoảng trống)
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')

def seed_milvus(URL_link: str, collection_name: str, filename: str, directory: str,use_ollama:bool=False):
    """
    Hàm tạo và lưu vectors embeddings vào Milvus từ dữ liệu local
    Args:
        URl_link: str: Đường dẫn đến Milvus server (ví dụ: "http://localhost:19530")
        collection_name: str: Tên collection trong Milvus để lưu dữ liệu
        filename: str: Tên file chứa dữ liệu
        directory: str: Đường dẫn đến thư mục chứa file dữ liệu
    Returns:
        Milvus : Đối tượng đã được khởi tạo chưa các vectors embeddings
    Chú ý:
        - Sử dụng model "text-embedding-3-large" để tạo embeddings
        - Collection cũ sẽ bị xóa nếu đã tồn tại (drop_collection=true)
    """
    # Khởi tạo model embeddings
    embeddings = _get_embeddings(use_ollama)
    # Đọc dữ liệu từ file local
    local_data, doc_name = load_data_from_local_file(filename, directory)
    #Chuyển đổi dữ liệu thành danh sách các Document với giá trị mặc định cho các trường
    documents = [
        Document(
            page_content=doc.get("page_content") or "",
            metadata = {
                "source": doc['metadata'].get("source", doc_name),
                'content_type': doc['metadata'].get("content_type") or 'text/plain',
                'title': doc['metadata'].get("title") or "",
                'description': doc['metadata'].get("description") or "",
                'language': doc['metadata'].get("language") or "en",
                'doc_name': doc_name, # biết được file nào để dễ xóa
                'start_index': doc['metadata'].get("start_index") or 0,
                }
            )
            for doc in local_data
        ]
    print(f'Loaded {len(documents)} documents')
    
    # Tạo id duy nhất cho mỗi document
    uuids = [str(uuid4()) for _ in range(len(documents))]
    
    # Xóa collection cũ trước (workaround bug drop_old trong langchain-milvus 0.3.x)
    _drop_collection_if_exists(URL_link, collection_name)
    
    # Khởi tạo Milvus vectorstore (không dùng drop_old=True để tránh bug)
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URL_link},
        collection_name=collection_name,
        drop_old=False
    )
    # Đăng ký ORM connection để tránh ConnectionNotExistException
    _ensure_orm_connection(vectorstore)
    # Thêm documents vào vectorstore
    vectorstore.add_documents(documents, ids=uuids)
    print('vector:', vectorstore)
    return vectorstore

def seed_milvus_live(url: str, URL_link: str, collection_name: str, doc_name: str,use_ollama:bool=False )-> Milvus:
    """
    Hàm crawl dữ liệu trực tiếp từ URL và lưu vectors embeddings vào Milvus
    Args:
        url: str: URL cần crawl dữ liệu
        URl_link: str: Đường dẫn đến Milvus server (ví dụ: "http://localhost:19530")
        collection_name: str: Tên collection trong Milvus để lưu dữ liệu
        doc_name: str: Tên tài liệu để lưu trong metadata của các document
    Returns:
        Milvus : Đối tượng đã được khởi tạo chưa các vectors embeddings
    Chú ý:
        - Sử dụng hàm crawl_web để lấy dữ liệu từ URL
        - Tự động gán metadata mặc định cho các trường thiếu
    """
    # Khởi tạo model embeddings
    embeddings = _get_embeddings(use_ollama)

    # Đọc dữ liệu từ URL trực tiếp
    documents = crawl_web(url)
    
    #cập nhật metadata cho mỗi document với giá trị mặc định
    for doc in documents:
        metadata = {
            'source': doc.metadata.get("source") or "",
            'content_type': doc.metadata.get("content_type") or 'text/plain',
            'title': doc.metadata.get("title") or "",
            'description': doc.metadata.get("description") or "",
            'language': doc.metadata.get("language") or "en",
            'doc_name': doc_name, # biết được file nào để dễ xóa
            'start_index': doc.metadata.get("start_index") or 0,
            
        }
        doc.metadata.update(metadata)
    
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Xóa collection cũ trước (workaround bug drop_old trong langchain-milvus 0.3.x)
    _drop_collection_if_exists(URL_link, collection_name)
    
    # Khởi tạo Milvus vectorstore (không dùng drop_old=True để tránh bug)
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URL_link},
        collection_name=collection_name,
        drop_old=False
    )
    # Đăng ký ORM connection để tránh ConnectionNotExistException
    _ensure_orm_connection(vectorstore)
    # Thêm documents vào Milvus
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector:', vectorstore)
    return vectorstore

def connect_to_milvus(URL_link: str, collection_name: str,use_ollama=False) -> Milvus:
    """
    Hàm kết nối đến colection đã có sẵn trong Milvus
    Args:
        URl_link: str: Đường dẫn đến Milvus server (ví dụ: "http://localhost:19530")
        collection_name: str: Tên collection trong Milvus để kết nối
    Returns:
        Milvus : Đối tượng đã được kết nối, sẵn sàng để truy vấn
    Chú ý:
        - Không tạo collection mới hoặc xóa dữ liệu cũ
        - sử dụng model "nomic-embed-text" và "sentence-transformers/all-MiniLM-L6-v2" để tạo embeddings khi truy vấn
        
    """
    embeddings = _get_embeddings(use_ollama)

    # Khởi tạo cấu hình Milvus
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URL_link},
        collection_name=collection_name
    )
    # Đăng ký ORM connection để tránh ConnectionNotExistException
    _ensure_orm_connection(vectorstore)
    return vectorstore
def main():
    """
    Hàm chính để kiểm thử các chức năng của module
    Thực hiện:
        1. Test seed_data với dữ liệu file từ local 'stack.json'
        2. (Đã commeemt) Test seed_milvus_live với dữ liệu từ URL 'http://www.stack-ai.com/docs'
    Chú ý:
        - Đảm bảo Milvus server đang chạy tại localhost:19530
        - Các biến môi trường cần thiết đã được cấu hình (ví dụ: OPENAI_API_KEY)
    """
    
    # Test seed_data với dữ liệu file từ local 'stack.json'
    seed_milvus("http://localhost:19530", "data_test", "stack_ai.json", "data",use_ollama=False)
    # Test seed_milvus_live với dữ liệu từ URL 'http://www.stack-ai.com/docs'       
    # seed_milvus_live("http://www.stack-ai.com/docs", "http://localhost:19530", "data_test_live_v2", "stack_ai")

#Chạy main () nếu file được thực thi trực tiếp 
if __name__ == "__main__":
    main()
    