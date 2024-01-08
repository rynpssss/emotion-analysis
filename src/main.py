import datetime
import pickle

import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from src import data
from src.models import predict_model
from src import models

# 学習モデルパス
TOKEN_PATH = models.TOKEN_PATH
MODEL_PATH = models.MODEL_PATH
STANDARD_SCALER_PATH = models.STANDARD_SCALER_PATH
PCA_PATH = models.PCA_PATH
MINMAX_SCALER_PATH = models.MINMAX_SCALER_PATH


def main():
    """ 文章に対する感情分析とモチベーションスコアの出力"""

    # 予測する日記の日付
    today = datetime.datetime.today().strftime('%Y%m%d')

    # ==========================================================================
    # 読み込み

    # 学習モデルの読み込み
    tokenizer = AutoTokenizer.from_pretrained(TOKEN_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # 予測用データの読み込み（日々の日記）
    # txt_path = f'./data/diary/diary_{today}.txt'
    txt_path = './data/diary/diary_231228.txt'
    sentence = data.read_txt(txt_path)

    # ==========================================================================
    # 予測

    # 感情予測
    emotion = predict_model.analyze_emotion(model, tokenizer, sentence)
    emotion_df = pd.DataFrame(emotion.values(), index=emotion.keys()).T

    # モチベーションスコア
    standard_scaler = pickle.load(open(STANDARD_SCALER_PATH, 'rb'))
    pca = pickle.load(open(PCA_PATH, 'rb'))
    minmax_scaler = pickle.load(open(MINMAX_SCALER_PATH, 'rb'))

    # スコアリング (主成分はPC1のみを使用してスコア化　※PC1がポジネガの成分を含んでいたため)
    # 標準化→PCA→正規化してスコア化しているが、全てtransfomersと同データにて学習済み

    scaled_x = standard_scaler.transform(emotion_df)

    pca_df = pd.DataFrame(pca.transform(scaled_x))
    pca_df = pca_df.rename(columns=pca.rename_pc_cols)
    pca_df = pca_df[['PC1']]
    pca_df['PC1'] = pca_df['PC1'] * -1

    motivation_score = minmax_scaler.transform(pca_df)
    emotion_df['モチベーションスコア'] = motivation_score

    # ==========================================================================
    # 保存
    temprate_output_path = './data/interim/date_diary_emotion.csv'
    output_path = temprate_output_path.replace('date', today)
    data.output_csv(emotion_df, output_path)


if __name__ == '__main__':
    main()
