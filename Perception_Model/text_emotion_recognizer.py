# 基于论文 2024.acl-long.801 实现的文本情感识别主代码
# 仅处理文本情感，不包含语音
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
from typing import Union, List, Dict

# ====================== 论文对齐配置 ======================
EMOTION_LABELS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]
MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5
MAX_TEXT_LENGTH = 128
# ==========================================================

# 模型初始化
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTION_LABELS)
).to(DEVICE)

def text_emotion_recognition(
    text: Union[str, List[str]]
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    论文核心：文本情感识别推理函数
    :param text: 输入文本（单句/多句）
    :return: 情感标签 + 置信度
    """
    is_single = isinstance(text, str)
    text = [text] if is_single else text

    # 文本编码
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    # 模型推理（无梯度，标准测试模式）
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # 解析结果
    results = []
    for prob in probs:
        idx = np.argmax(prob)
        score = round(float(prob[idx]), 4)
        label = EMOTION_LABELS[idx] if score >= CONFIDENCE_THRESHOLD else "unknown"
        results.append({label: score})

    return results[0] if is_single else results

# ====================== 项目API入口 ======================
def get_text_emotion_api(text: Union[str, List[str]]):
    """
    
    """
    return text_emotion_recognition(text)