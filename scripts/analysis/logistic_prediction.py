"""
ロジスティックモデルによる将来予測
"""
from typing import Tuple, Callable
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


def predict_future(
    t_actual: np.ndarray, 
    P_actual: np.ndarray, 
    gamma: float, 
    K: float, 
    forecast_end_t: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ロジスティックモデルを使用して将来予測を行う
    
    Args:
        t_actual (np.ndarray): 実績データの期間
        P_actual (np.ndarray): 実績データの値
        gamma (float): 成長率パラメータ
        K (float): 環境収容力
        forecast_end_t (int): 予測の最終期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (予測時刻の配列, 予測値の配列)
    """
    P0: float = P_actual[0]
    t_start_forecast: float = t_actual[0]
    dt_forecast: float = 1.0

    # 将来予測の計算
    t_forecast: np.ndarray
    P_forecast: np.ndarray
    t_forecast, P_forecast = runge_kutta_4th_order(
        P0, t_start_forecast, forecast_end_t, dt_forecast, gamma, K, logistic_model
    )
    
    return t_forecast, P_forecast
