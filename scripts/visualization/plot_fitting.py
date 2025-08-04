"""
ロジスティックモデルのフィッティング結果をプロット
"""
import matplotlib.pyplot as plt
from scripts.analysis.core.logistic_model import logistic_model
from scripts.analysis.core.numerical_methods import runge_kutta_4th_order

# 日本語フォントの設定 (macOS標準のヒラギノ角ゴシック)
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け対策

def plot_fit_result(t_actual, P_actual, best_params, output_path):
    """
    最終的なモデルを計算し、比較グラフをプロットする
    
    Args:
        t_actual (np.array): 実績データの時刻
        P_actual (np.array): 実績データの人口値
        best_params (dict): 最適化されたパラメータ {"gamma": float, "K": float}
        output_path (str): 出力画像のファイルパス
    """
    P0 = P_actual[0]
    final_gamma = best_params["gamma"]
    final_K = best_params["K"]

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
