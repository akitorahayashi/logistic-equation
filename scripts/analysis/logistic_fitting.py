"""
ロジスティックモデルのパラメータフィッティング
"""
import pandas as pd
import numpy as np
import sys

from .core.logistic_model import logistic_model
from .core.numerical_methods import runge_kutta_4th_order

def load_data(xlsx_path, header=0):
    """xlsxファイルからtime, value列を抽出"""
    try:
        df = pd.read_excel(xlsx_path, header=header)
        if "time" in df.columns and "value" in df.columns:
            t_actual = df["time"].values
            P_actual = df["value"].values
            return t_actual, P_actual
        else:
            print("エラー: Excelファイルに 'time' および 'value' 列が必要です。")
            sys.exit(1)
    except FileNotFoundError:
        print(f"エラー: '{xlsx_path}' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"Excelファイルの読み込みエラー: {e}")
        sys.exit(1)

def find_best_params(t_actual, P_actual, K_range, gamma_range):
    """
    残差平方和(SSE)を最小化することで、最適なγとKのパラメータを見つけ出す
    
    Args:
        t_actual (np.array): 実績データの時刻
        P_actual (np.array): 実績データの人口値
        K_range (np.array): 環境収容力Kの探索範囲
        gamma_range (np.array): 成長率γの探索範囲
    
    Returns:
        tuple: (最適パラメータ辞書, 最小SSE値)
    """
    P0 = P_actual[0]
    best_params = {"gamma": 0, "K": 0}
    min_sse = np.inf

    for K in K_range:
        for gamma in gamma_range:
            t_model, P_model = runge_kutta_4th_order(P0, t_actual[0], t_actual[-1], 1.0, gamma, K, logistic_model)
            # SSE計算のために、実績データ点に対応するモデル上の値を内挿
            model_P_at_actual_t = np.interp(t_actual, t_model, P_model)
            sse = np.sum((P_actual - model_P_at_actual_t)**2)
            if sse < min_sse:
                min_sse = sse
                best_params["gamma"] = gamma
                best_params["K"] = K
    return best_params, min_sse
