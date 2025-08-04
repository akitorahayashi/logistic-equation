"""
ロジスティックモデルのパラメータフィッティング
"""
import sys
from typing import Tuple, Dict, Callable
import pandas as pd
import numpy as np


def logistic_model(t: float, P: float, gamma: float, K: float) -> float:
    """
    ロジスティック方程式: dP/dt = gamma * P * (1 - P/K)
    
    Args:
        t (float): 時間
        P (float): 現在の人口
        gamma (float): 成長率
        K (float): 環境収容力
    
    Returns:
        float: 人口の変化率 dP/dt
    """
    return gamma * P * (1 - P / K)


def runge_kutta_4th_order(
    P0: float, 
    t_start: float, 
    t_end: float, 
    dt: float, 
    gamma: float, 
    K: float, 
    model_func: Callable[[float, float, float, float], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    4次ルンゲ・クッタ法による数値積分
    
    Args:
        P0 (float): 初期値
        t_start (float): 開始時刻
        t_end (float): 終了時刻
        dt (float): 時間刻み
        gamma (float): 成長率
        K (float): 環境収容力
        model_func: モデル関数
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (時刻配列, 人口配列)
    """
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)
    P = np.zeros(n_steps)
    P[0] = P0
    
    for i in range(1, n_steps):
        k1 = dt * model_func(t[i-1], P[i-1], gamma, K)
        k2 = dt * model_func(t[i-1] + dt/2, P[i-1] + k1/2, gamma, K)
        k3 = dt * model_func(t[i-1] + dt/2, P[i-1] + k2/2, gamma, K)
        k4 = dt * model_func(t[i-1] + dt, P[i-1] + k3, gamma, K)
        P[i] = P[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, P


def load_data(xlsx_path: str, header: int = 0) -> Tuple[np.ndarray, np.ndarray]:
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


def find_best_params(
    t_actual: np.ndarray, 
    P_actual: np.ndarray, 
    K_range: np.ndarray, 
    gamma_range: np.ndarray
) -> Tuple[Dict[str, float], float]:
    """
    残差平方和(SSE)を最小化することで、最適なγとKのパラメータを見つけ出す
    
    Args:
        t_actual (np.ndarray): 実績データの時刻
        P_actual (np.ndarray): 実績データの人口値
        K_range (np.ndarray): 環境収容力Kの探索範囲
        gamma_range (np.ndarray): 成長率γの探索範囲
    
    Returns:
        Tuple[Dict[str, float], float]: (最適パラメータ辞書, 最小SSE値)
    """
    P0: float = P_actual[0]
    best_params: Dict[str, float] = {"gamma": 0.0, "K": 0.0}
    min_sse: float = np.inf

    for K in K_range:
        for gamma in gamma_range:
            t_model, P_model = runge_kutta_4th_order(P0, t_actual[0], t_actual[-1], 1.0, gamma, K, logistic_model)
            # SSE計算のために、実績データ点に対応するモデル上の値を内挿
            model_P_at_actual_t: np.ndarray = np.interp(t_actual, t_model, P_model)
            sse: float = np.sum((P_actual - model_P_at_actual_t)**2)
            if sse < min_sse:
                min_sse = sse
                best_params["gamma"] = gamma
                best_params["K"] = K
    return best_params, min_sse
