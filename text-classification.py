import pandas as pd
import os
import logging
from datetime import datetime

# 配置日志
log_filename = f'book_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 设置代理
logger.info("Setting up proxy configuration")
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"

# 读取数据
logger.info("Reading books_cleaned.csv file")
try:
    books = pd.read_csv("books_cleaned.csv")
    logger.info(f"Successfully loaded {len(books)} books from CSV")
except Exception as e:
    logger.error(f"Error reading CSV file: {e}")
    raise

# 分类映射
logger.info("Setting up category mapping")
category_mapping = {
    'Fiction': "Fiction",
    'Juvenile Fiction': "Children's Fiction",
    'Biography & Autobiography': "Nonfiction",
    'History': "Nonfiction",
    'Literary Criticism': "Nonfiction",
    'Philosophy': "Nonfiction",
    'Religion': "Nonfiction",
    'Comics & Graphic Novels': "Fiction",
    'Drama': "Fiction",
    'Juvenile Nonfiction': "Children's Nonfiction",
    'Science': "Nonfiction",
    'Poetry': "Fiction"
}

logger.info("Mapping categories")
books["simple_categories"] = books["categories"].map(category_mapping)

# 加载模型
logger.info("Loading zero-shot classification pipeline")
from transformers import pipeline
fiction_categories = ["Fiction", "Nonfiction"]
try:
    pipe = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli",
                   device="cpu")
    logger.info("Successfully loaded classification pipeline")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")
    raise

def generate_predictions(sequence, categories):
    try:
        predictions = pipe(sequence, categories)
        max_index = np.argmax(predictions["scores"])
        max_label = predictions["labels"][max_index]
        return max_label
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return None

# 生成预测
logger.info("Starting predictions for Fiction category")
from tqdm import tqdm
import numpy as np

actual_cats = []
predicted_cats = []

try:
    for i in tqdm(range(0, 300)):
        sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[i]
        pred = generate_predictions(sequence, fiction_categories)
        predicted_cats.append(pred)
        actual_cats.append("Fiction")
    
    logger.info("Starting predictions for Nonfiction category")
    for i in tqdm(range(0, 300)):
        sequence = books.loc[books["simple_categories"] == "Nonfiction", "description"].reset_index(drop=True)[i]
        pred = generate_predictions(sequence, fiction_categories)
        predicted_cats.append(pred)
        actual_cats.append("Nonfiction")
except Exception as e:
    logger.error(f"Error during prediction generation: {e}")
    raise

# 创建预测数据框
logger.info("Creating predictions dataframe")
predictions_df = pd.DataFrame({"actual_categories": actual_cats, "predicted_categories": predicted_cats})

# 计算准确率
predictions_df["correct_prediction"] = (
    np.where(predictions_df["actual_categories"] == predictions_df["predicted_categories"], 1, 0)
)
accuracy = predictions_df["correct_prediction"].sum() / len(predictions_df)
logger.info(f"Model accuracy: {accuracy:.2f}")

# 处理缺失分类
logger.info("Processing missing categories")
isbns = []
predicted_cats = []

missing_cats = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)
logger.info(f"Found {len(missing_cats)} books with missing categories")

try:
    for i in tqdm(range(0, len(missing_cats))):
        sequence = missing_cats["description"][i]
        pred = generate_predictions(sequence, fiction_categories)
        predicted_cats.append(pred)
        isbns.append(missing_cats["isbn13"][i])
except Exception as e:
    logger.error(f"Error processing missing categories: {e}")
    raise

# 合并预测结果
logger.info("Merging predictions with original dataset")
missing_predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})
books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(books["simple_categories"].isna(), books["predicted_categories"], books["simple_categories"])
books = books.drop(columns=["predicted_categories"])

# 保存结果
logger.info("Saving final results to CSV")
try:
    books.to_csv("books_with_categories.csv", index=False)
    logger.info("Successfully saved results to books_with_categories.csv")
except Exception as e:
    logger.error(f"Error saving results to CSV: {e}")
    raise

logger.info("Processing completed successfully")