"""
ロジスティックモデルのフィッティング結果をプロット
"""
from typing import Dict, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォントの設定 (macOS標準のヒラギノ角ゴシック)
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け対策


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


def plot_fit_result(
    t_actual: np.ndarray, 
    P_actual: np.ndarray, 
    best_params: Dict[str, float], 
    output_path: str
) -> None:
    """
    最終的なモデルを計算し、比較グラフをプロットする
    
    Args:
        t_actual (np.ndarray): 実績データの時刻
        P_actual (np.ndarray): 実績データの人口値
        best_params (Dict[str, float]): 最適化されたパラメータ {"gamma": float, "K": float}
        output_path (str): 出力画像のファイルパス
    """
    P0: float = P_actual[0]
    final_gamma: float = best_params["gamma"]
    final_K: float = best_params["K"]

    t_final_model: np.ndarray
    P_final_model: np.ndarray
    t_final_model, P_final_model = runge_kutta_4th_order(P0, t_actual[0], t_actual[-1], 1.0, final_gamma, final_K, logistic_model)

    plt.figure(figsize=(10, 6))
    plt.plot(t_actual, P_actual, 'o', label='実データ')
    plt.plot(t_final_model, P_final_model, '-', label=f'ロジスティックモデル (γ={final_gamma:.4f}, K={final_K})')
    plt.title('実データとロジスティックモデルの比較')
    plt.xlabel('時間')
    plt.ylabel('値')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"グラフを '{output_path}' として保存しました。")
    plt.close() # メモリリークを防ぐためにプロットを閉じる
