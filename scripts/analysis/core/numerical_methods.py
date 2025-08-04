"""
数値計算手法の実装
"""
import numpy as np

def runge_kutta_4th_order(P0, t_start, t_end, dt, gamma, K, model_func):
    """
    4次ルンゲクッタ法によるロジスティック方程式の数値解法
    
    Args:
        P0 (float): 初期値
        t_start (float): 開始時刻
        t_end (float): 終了時刻
        dt (float): 時間刻み幅
        gamma (float): 成長率パラメータ
        K (float): 環境収容力
        model_func (callable): 微分方程式を表す関数
    
    Returns:
        tuple: (時刻の配列, 人口値の配列)
    """
    t_values = np.arange(t_start, t_end + dt, dt)
    P_values = np.zeros(len(t_values))
    P_values[0] = P0

    for i in range(len(t_values) - 1):
        P = P_values[i]
        k1 = dt * model_func(P, gamma, K)
        k2 = dt * model_func(P + 0.5 * k1, gamma, K)
        k3 = dt * model_func(P + 0.5 * k2, gamma, K)
        k4 = dt * model_func(P + k3, gamma, K)
        P_values[i+1] = P + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t_values, P_values
