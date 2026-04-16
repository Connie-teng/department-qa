# 國立中正大學資管系 AI 問答助理

## 專案簡介
基於 RAG（檢索增強生成）架構的系所問答 AI 系統，能回答學生關於課程規定、教師介紹、表單下載等問題。

## 技術架構
- **前端介面**：Streamlit
- **語言模型**：Gemini 2.5 Flash API / LM Studio（本機）
- **向量資料庫**：FAISS
- **Embedding**：sentence-transformers
- **框架**：LangChain
- **資料來源**：網頁爬蟲自動抓取系所網站

## 功能特色
- 自然語言查詢系所資訊
- 自動爬取並索引系所網站內容
- 涵蓋教師介紹、修業規定、表單下載、招生資訊、考古題等
- 回答附帶參考來源

## 系統架構圖
使用者輸入問題 → Embedding 向量化 → FAISS 相似度搜尋 → LLM 生成回答

## 使用方式
1. 安裝套件：`pip install -r requirements.txt`
2. 爬取資料：`python crawler.py`
3. 建立索引：`python ingest.py`
4. 啟動介面：`streamlit run app.py`

## Demo 影片
👉 [點此觀看系統操作示範](https://youtu.be/F7OliXXW5pg)
