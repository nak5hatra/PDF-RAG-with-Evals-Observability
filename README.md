# 📄 PDF RAG with Evals Observability

🚀 A **production-ready Retrieval-Augmented Generation (RAG) pipeline** for querying PDFs, built with:  
- **LangChain + LangGraph** → modular RAG pipeline  
- **ChromaDB** → vector storage  
- **OpenAI GPT-5** → LLM reasoning  
- **FastAPI** → REST API service  
- **LangSmith** → observability (latency, traces)  
- **Docker** → scalable cloud deployment  

---

## ✨ Features
- 
- 🔎 **Ask questions** against ingested PDFs via `/ask`  
- 📊 **Observability** → latency 
- ☁️ **Cloud-native** → containerized and deployed
- 🔐 **Secure & modular** design for enterprise-ready workflows  

---

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python 3.11  
- **LLM:** OpenAI (GPT-5) via LangChain  
- **Vector Store:** ChromaDB  
- **Observability:** LangSmith + custom logs  
- **Deployment:** Docker, GCP Cloud Run  

---

## 🚀 Getting Started

### 1️⃣ Clone repo
```bash
git clone https://github.com/nak5hatra/PDF-RAG-with-Evals-Observability.git
cd PDF-RAG-with-Evals-Observability
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run locally
```bash
uvicorn src.api:app --reload --port 8000
```
```bash
Visit 👉 http://127.0.0.1:8000/docs
```
---

## 🐳 Docker Deployment

```
sudo docker build -t pdf-rag-api .
```
```
sudo docker run -p 8000:8000 pdf-rag-api
```

### 🧪 Future Work

- AWS/GCP Deployment
- Persist vector DB in GCS / Postgres for scaling
- Add frontend UI (Streamlit / React)

### 👤 Author

Nakshatra Sharma
[🔗 Linkedin](https://www.linkedin.com/in/nak5hatra/)   
💼 Data Analyst → Aspiring AI Engineer