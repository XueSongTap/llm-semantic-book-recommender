import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
import os
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
# 读取数据
print("Loading books data...")
books = pd.read_csv("books_with_categories.csv")

# 初始化情感分析模型
print("Initializing emotion classifier...")
classifier = pipeline("text-classification",
                     model="j-hartmann/emotion-english-distilroberta-base",
                     top_k=None,
                     device="cpu")

# 定义情感标签
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

def calculate_max_emotion_scores(predictions):
    """计算每种情感的最高分数"""
    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

# 初始化存储结构
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

# 处理所有书籍描述
print("Analyzing emotions in book descriptions...")
for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])

# 创建情感得分数据框
print("Creating emotions dataframe...")
emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn

# 合并数据
print("Merging data...")
books = pd.merge(books, emotions_df, on="isbn13")

# 保存结果
print("Saving results...")
books.to_csv("books_with_emotions.csv", index=False)
print("Analysis completed!")