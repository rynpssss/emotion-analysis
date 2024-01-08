# import datetime

# import pandas as pd
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification

# from src import data
# from src.models import predict_model


# # 一括実行
# def main():

#     # 予測する日記の日付
#     today = datetime.datetime.today().strftime('%Y%m%d')

#     # 学習モデルの読み込み
#     checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
#                                                                num_labels=8)

#     # 予測用データの読み込み（日々の日記）
#     txt_path = f'../data/diary/diary_{today}.txt'
#     with open(txt_path, 'r', encoding='utf-8') as file:
#         sentence = file.read()

#     # 感情予測
#     result = predict_model.analyze_emotion(model, tokenizer, sentence)
#     result_df = pd.DataFrame(result.values(), index=result.keys()).T
#     result_df['Sentence'] = sentence

#     # 保存
#     temprate_output_path = data.OUTPTU_DATA_PATH
#     output_path = temprate_output_path.replace('date', today)
#     data.output_csv(result_df, output_path)


# if __name__ == '__main__':
#     main()
