import pandas as pd
import numpy as np


def delete_index_under_num(df, cols, under):
    """ 指定数字以下のindexを削除する"""

    df['max'] = df[cols].max(axis=1)
    df = df[df['max'] >= under]
    df = df.drop('max', axis=1)

    return df


def create_answer_col(df, cols):
    """ 正解データの作成（感情系カラムに対する正規化リスト）"""

    df['emotions'] = df.apply(lambda x: [x[c] for c in cols], axis=1)
    df['answer'] = df['emotions'].apply(lambda x: [xi/np.sum(x) for xi in x])

    return df
