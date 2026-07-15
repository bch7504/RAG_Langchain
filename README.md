# 🤖 Chatbot AI với LangChain và Python

Ứng dụng chatbot AI sử dụng **RAG (Retrieval-Augmented Generation)** kết hợp **LangChain**, **Milvus Vector Database** và các mô hình ngôn ngữ lớn (LLM) để trả lời câu hỏi dựa trên dữ liệu được cung cấp.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green?logo=chainlink&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-v3.0--beta-purple?logo=apache&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Required-blue?logo=docker&logoColor=white)

---

## 📑 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Các bước cài đặt và chạy](#-các-bước-cài-đặt-và-chạy)
- [Cách sử dụng](#-cách-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## ✨ Tính năng

- 🔍 **RAG Pipeline** — Tìm kiếm và trả lời dựa trên dữ liệu thực tế với Ensemble Retriever (70% Vector Search + 30% BM25)
- 🌐 **Web Crawling** — Tự động crawl dữ liệu từ website với độ sâu tối đa 4 cấp
- 📄 **Hỗ trợ JSON local** — Tải và xử lý dữ liệu từ file JSON có sẵn
- 🧠 **Đa mô hình LLM** — Hỗ trợ cả Google Gemini và Ollama (LLaMA2, Qwen2.5)
- 🔄 **Đa mô hình Embedding** — Chọn giữa Ollama (`nomic-embed-text`) hoặc HuggingFace (`all-MiniLM-L6-v2`)
- 💬 **Giao diện chat** — UI thân thiện với Streamlit, hỗ trợ lịch sử hội thoại
- 🗄️ **Vector Database** — Lưu trữ và truy vấn vector embeddings với Milvus
- 📊 **Quản lý dữ liệu** — Xem và quản lý collections thông qua Attu UI

---

## 🏗️ Kiến trúc hệ thống

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit   │────▶│  LangChain Agent │────▶│   LLM Provider  │
│  (Frontend)  │     │  (Orchestrator)  │     │ Gemini / Ollama  │
└──────────────┘     └────────┬─────────┘     └─────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Ensemble Retriever │
                    │  (70% + 30%)       │
                    ├────────┬───────────┤
                    │ Milvus │   BM25    │
                    │ Vector │   Text    │
                    │ Search │  Search   │
                    └────────┴───────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Milvus Database  │
                    │   (Docker)         │
                    └────────────────────┘
```

---

## 📋 Yêu cầu hệ thống

| Yêu cầu | Chi tiết |
|----------|----------|
| **Python** | 3.8 trở lên, khuyến nghị **3.8.18** ([Tải tại đây](https://www.python.org/downloads/)) |
| **Docker Desktop** | Phiên bản mới nhất ([Tải tại đây](https://www.docker.com/products/docker-desktop/)) |
| **Google API Key** | Để sử dụng Gemini LLM ([Đăng ký tại đây](https://aistudio.google.com/apikey)) |
| **RAM** | Khoảng **4GB RAM** trống |
| **Ollama** *(tùy chọn)* | Nếu muốn sử dụng LLM/Embedding offline ([Tải tại đây](https://ollama.com/download)) |

---

## 🚀 Các bước cài đặt và chạy

### Bước 1: Cài đặt môi trường

Khuyến nghị dùng **Conda** để quản lý môi trường Python:

```bash
# Tạo môi trường mới với Python 3.8.18
conda create -n myenv python=3.8.18

# Kích hoạt môi trường
conda activate myenv
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Bước 2: Tải xuống Ollama *(tùy chọn — nếu muốn dùng LLM/Embedding offline)*

1. Truy cập: https://ollama.com/download
2. Chọn phiên bản phù hợp với hệ điều hành
3. Cài đặt theo hướng dẫn
4. Tải mô hình cần thiết:

```bash
# LLM model
ollama run qwen2.5:7b

# Embedding model
ollama pull nomic-embed-text
```

### Bước 3: Cài đặt và chạy Milvus Database

1. Khởi động **Docker Desktop**
2. Mở Terminal/Command Prompt, di chuyển vào thư mục `volumes` và chạy:

```bash
cd volumes
docker compose up --build
```

Sau khi chạy thành công, các service sau sẽ hoạt động:

| Service | Port | Mô tả |
|---------|------|--------|
| Milvus | `19530` | Vector database server |
| MinIO | `9000` / `9001` | Object storage |
| etcd | `2379` | Metadata storage |
| Attu | `3000` | Milvus Web UI |

> 💡 **Tip:** Truy cập **Attu UI** tại http://localhost:3000 để xem và quản lý dữ liệu trong Milvus.

### Bước 4: Cấu hình API Keys

Tạo file `.env` trong thư mục gốc của dự án:

```env
# Google Gemini API Key (bắt buộc nếu dùng Gemini)
GOOGLE_API_KEY=your-google-api-key-here
```

**Tùy chọn — Cấu hình LangSmith** (để theo dõi và debug):

Truy cập [LangSmith](https://smith.langchain.com/) để lấy API key, sau đó thêm vào `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langchain-api-key-here"
LANGCHAIN_PROJECT="project-name"
```

### Bước 5: Chạy ứng dụng

**1. Crawl dữ liệu về local:**

```bash
python crawl.py
```

**2. Seed dữ liệu vào Milvus:**

```bash
python seed_data.py
```

> 📌 Kiểm tra dữ liệu đã được seed thành công bằng cách truy cập **Attu UI**: http://localhost:3000

**3. Chạy ứng dụng chatbot:**

```bash
streamlit run main.py
```

---

## 💻 Cách sử dụng

### 1. Khởi động ứng dụng

- ✅ Đảm bảo **Docker Desktop** đang chạy (Milvus cần Docker)
- ✅ Đảm bảo **Ollama** đang chạy nếu bạn chọn sử dụng Ollama
- ✅ Chạy lệnh: `streamlit run main.py`

### 2. Tải và xử lý dữ liệu

#### Cách 1: Từ file JSON local

1. Chọn tab **"File local"** ở thanh bên
2. Nhập tên collection trong Milvus (mặc định: `data_test`)
3. Nhập tên file JSON (mặc định: `stack_ai.json`)
4. Nhập thư mục chứa file (mặc định: `data`)
5. Nhấn **"Tải dữ liệu từ file"**
6. Đợi hệ thống xử lý và thông báo thành công

#### Cách 2: Từ URL trực tiếp

1. Chọn tab **"URL trực tiếp"** ở thanh bên
2. Nhập URL cần crawl dữ liệu
3. Nhập tên collection (mặc định: `data_test_live`)
4. Nhấn **"Crawl dữ liệu"**
5. Đợi hệ thống crawl và xử lý dữ liệu

### 3. Tương tác với chatbot

1. Nhập câu hỏi vào ô chat ở phần dưới màn hình
2. Nhấn **Enter** để gửi câu hỏi
3. Chatbot sẽ:
   - 🔍 Tìm kiếm thông tin liên quan trong cơ sở dữ liệu (Ensemble Retriever)
   - 🔄 Kết hợp kết quả từ Vector Search và BM25
   - 💡 Tạo câu trả lời dựa trên ngữ cảnh tìm được
4. Lịch sử chat được hiển thị và duy trì trong phiên làm việc

### 4. Xem thông tin hệ thống

- Theo dõi cấu hình **Embedding model** và **LLM model** ở thanh bên
- Chọn **collection** để truy vấn
- Nhấn **🔄 Refresh Agent** sau khi tải dữ liệu mới để cập nhật agent

---

## 📂 Cấu trúc dự án

```
practice/
├── main.py              # Entry point — giao diện Streamlit
├── agent.py             # Cấu hình LLM Agent và Retriever
├── crawl.py             # Web crawling và xử lý dữ liệu
├── seed_data.py         # Seed dữ liệu vào Milvus vector database
├── requirements.txt     # Danh sách dependencies
├── .env                 # API keys (không commit lên Git)
├── .gitignore           # Các file/thư mục bỏ qua khi commit
├── data/                # Thư mục chứa dữ liệu JSON đã crawl
├── data_v3/             # Thư mục chứa dữ liệu phiên bản 3
└── volumes/
    └── docker-compose.yml  # Docker Compose cho Milvus stack
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản |
|------------|-----------|-----------|
| **Framework AI** | LangChain | 1.3.11 |
| **Agent Framework** | LangGraph | latest |
| **Vector Database** | Milvus | v3.0-beta |
| **LLM** | Google Gemini / Ollama | Gemini 2.5 Flash / Qwen2.5:7b |
| **Embeddings** | HuggingFace / Ollama | all-MiniLM-L6-v2 / nomic-embed-text |
| **Web UI** | Streamlit | latest |
| **Web Crawling** | BeautifulSoup4 + LangChain Loaders | latest |
| **Retriever** | Ensemble (Milvus + BM25) | — |
| **Containerization** | Docker Compose | — |

---

## 📝 License

This project is for educational purposes.

---

<p align="center">
  Made with ❤️ using LangChain & Python
</p>
