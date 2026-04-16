from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("載入文件中...")
loader = DirectoryLoader(
    "./docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8-sig"}
)
docs = loader.load()
print(f"共載入 {len(docs)} 份文件")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks = splitter.split_documents(docs)
print(f"切成 {len(chunks)} 個段落")

print("向量化中（第一次會下載模型，約 500MB）...")
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("完成！索引已儲存至 faiss_index/")
