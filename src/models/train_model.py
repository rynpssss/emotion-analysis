import pandas as pd
from sklearn.decomposition import PCA


class Pca:
    """ 自作PCAクラス（PCAの基本機能をクラス化）"""

    def __init__(self, X, cols):
        """ 初期設定

            Parameter:
                X: np.array 標準化後の行列
                cols: list 行列のカラム名
        """
        self.X = X
        self.cols = cols
        self.rename_pc_cols = {i: f"PC{i+1}" for i in range(len(cols))}

    def fit(self):
        """ PCA実行"""
        self.pca = PCA()
        self.pca.fit_transform(self.X)

    def loadings(self):
        """ カラム別の主成分量"""
        df = pd.DataFrame(self.pca.components_.T, index=self.cols)
        df = df.rename(columns=self.rename_pc_cols)
        return df

    def score(self):
        """ インデックス別の主成分量"""
        df = pd.DataFrame(self.pca.transform(self.X))
        df = df.rename(columns=self.rename_pc_cols)
        return df

    def contribution(self):
        """ カラム別の寄与度"""
        df = pd.DataFrame(self.pca.explained_variance_ratio_)
        df = df.rename(columns={0: 'contribution'})
        df['cumsum'] = df['contribution'].cumsum()
        return df
    
    def transform(self, x):
        """ 追加データのPCA"""
        return self.pca.transform(x)
