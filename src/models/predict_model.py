import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def analyze_emotion(model, tokenizer, text):
    # 推論モードを有効化
    model.eval()

    # 入力データ変換 + 推論
    tokens = tokenizer(text=text, truncation=True, return_tensors="pt")
    tokens.to(model.device)
    preds = model(**tokens)
    prob = softmax(preds.logits.cpu().detach().numpy()[0])

    emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
    out_dict = {n: p for n, p in zip(emotion_names_jp, prob)}

    return out_dict
