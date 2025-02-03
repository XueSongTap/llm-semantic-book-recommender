from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # 更新导入
from langchain_community.vectorstores import Chroma  # 更新导入
from dotenv import load_dotenv
import pandas as pd

print("Loading environment variables...")
load_dotenv()

# 指定持久化目录
persist_directory = "db"

# 首次运行时创建和存储向量数据库
def create_vector_db():
    print("Reading books data from CSV...")
    books = pd.read_csv("books_cleaned.csv")
    
    print("Saving tagged descriptions to text file...")
    books["tagged_description"].to_csv(
        "tagged_description.txt",
        sep="\n",
        index=False,
        header=False
    )

    print("Loading text documents...")
    raw_documents = TextLoader("tagged_description.txt").load()

    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    documents = text_splitter.split_documents(raw_documents)

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    print("Creating vector database...")
    db = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_directory  # 指定存储目录
    )
    return db

# 后续使用时直接加载已存储的数据库
def load_vector_db():
    print("Loading existing vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("Vector database loaded")
    return db

def format_results(docs):
    results = []
    for doc in docs:
        content = doc.page_content
        # 提取 ISBN (假设ISBN是内容的前13位)
        isbn = content.split()[0]
        # 提取描述（移除ISBN）
        description = ' '.join(content.split()[1:])
        
        result = {
            'isbn': isbn,
            'title': isbn_to_title.get(isbn, "Unknown Title"),
            'description': description[:200] + "..."  # 只显示前200个字符
        }
        results.append(result)
    return results

# 使用示例
if __name__ == "__main__":
    import os
    # 加载书籍数据以获取标题信息
    books = pd.read_csv("books_cleaned.csv")
    isbn_to_title = dict(zip(books['isbn10'], books['title']))
    
    # 检查数据库是否已存在
    if not os.path.exists(persist_directory):
        db = create_vector_db()
    else:
        db = load_vector_db()

    # 使用数据库进行查询
    query = "A book to teach children about nature"
    docs = db.similarity_search(query, k=10)
    results = format_results(docs)
    
    print(f"\nTop 5 recommendations for: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   ISBN: {result['isbn']}")
        print(f"   Description: {result['description']}")
        print()