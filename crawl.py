import os
import re
import json
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv

def bs4_extractor( html:str) -> str:
    """
    Hàm trích xuất nội dung văn bản từ HTML sử dụng BeautifulSoup
    Args:
        html: Chuỗi HTML cần xử lý
    Returns:
        str: Văn bản đã dược làm sạch, loại bỏ các thẻ HTML và khoảng trắng thừa
    """
    soup = BeautifulSoup(html, 'html.parser') # phân tích cú pháp HTML
    return re.sub(r'\n\n+',"\n\n", soup.text).strip() # Xóa các khoảng trắng thừa và dòng trống thừa
    
def crawl_web(url_data):
    #Tạo loader với độ sâu tối đa là 4 cấp
    loader = RecursiveUrlLoader(url=url_data,extractor=bs4_extractor, max_depth=4)
    docs = loader.load()# tải nội dung
    print('length:', len(docs)) # in số lượng tài liệu đã tải
    
    #chia nhỏ văn bản thành các đoạn 1000 ký tự với chồng lấp 500 ký tự
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        
    )
    all_splits = text_splitter.split_documents(docs)
    print('length_all_split: ', len(all_splits)) # in số lượng đoạn văn bản đã chia nhỏ
    return all_splits

def web_base_loader(url_data):
    """
    Hàm tải dữ liệu từ một URL dơn (không đệ quy) không chui vào các link con
    Args:
        url_data: str: URL cần tải nội dung
    Returns:
        list: Danh sách các Document đã được chia nhỏ
    """
    loader = WebBaseLoader(url_data)
    docs = loader.load()
    print('length:', len(docs)) # in số lượng tài liệu đã tải
    
    #chia nhỏ văn bản thành các đoạn 1000 ký tự với chồng lấp 500 ký tự
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def save_data_locally(documents, filename, directory):
    """
    Lưu danh sách document vào file JSON
    Args:
        documents: list: Danh sách các Document cần lưu
        filename: str: Tên file JSON cần lưu (vd: data.json)
        directory: str: Thư mục chứa file ( vd: data_v3)
    Returns:
        None: Hàm không trả về giá trị gì chỉ lưu và in thông báo
    """
    if not os.path.exists(directory):
        os.makedirs(directory) # tạo thư mục nếu chưa tồn tại
    file_path = os.path.join(directory, filename)
    
    # Chuyển đổi danh sách Document thành danh sách dict để lưu vào JSON
    data_to_save = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    print(f'Data saved to {file_path}')

def main():
    """
    Hàm chính điều khiển luồng chương trình
    1. Crawl dữ liệu từ web stack-ai
    2. Lưu dữ liệu vào file JSON
    3. In ra kết quả crawl để kiểm tra 
    """
    # Crawl dữ liệu từ web stack-ai
    data = crawl_web("https://www.stack-ai.com/docs")
    # Lưu dữ liệu vào thư mục data
    save_data_locally(data, "stack_ai.json", "data")
    print('data: ', data) # in dữ liệu được chạy trực tiếp

if __name__ == "__main__":
    main()