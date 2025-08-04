"""
ロジスティックモデルの最適パラメータを見つけるためのCLIツール
"""
import argparse
from typing import Dict, Tuple
import numpy as np
from scripts.analysis.logistic_fitting import load_data, find_best_params
from scripts.visualization.plot_fitting import plot_fit_result

def main() -> None:
    """
    コマンドライン実行用のメイン関数
    パラメータ探索とフィッティングのためのCLIを提供
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="ロジスティックモデルの最適パラメータを見つけるためのCLIツール")
    parser.add_argument('--input', type=str, required=True, help="入力Excelファイルパス")
    parser.add_argument('--output', type=str, default="fit_result.png", help="出力PNGファイル名")
    parser.add_argument('--K_min', type=float, default=1000, help="Kの探索最小値")
    parser.add_argument('--K_max', type=float, default=100000, help="Kの探索最大値")
    parser.add_argument('--K_step', type=float, default=1000, help="Kの刻み幅")
    parser.add_argument('--gamma_min', type=float, default=0.01, help="γの探索最小値")
    parser.add_argument('--gamma_max', type=float, default=0.5, help="γの探索最大値")
    parser.add_argument('--gamma_step', type=float, default=0.001, help="γの刻み幅")
    parser.add_argument('--header', type=int, default=0, help="Excelのヘッダー行番号 (0始まり)")
    args: argparse.Namespace = parser.parse_args()

    # データの読み込み
    t_actual: np.ndarray
    P_actual: np.ndarray
    t_actual, P_actual = load_data(args.input, header=args.header)

    # パラメータ探索範囲の定義
    K_range: np.ndarray = np.arange(args.K_min, args.K_max, args.K_step)
    gamma_range: np.ndarray = np.arange(args.gamma_min, args.gamma_max, args.gamma_step)

    # 最適パラメータの探索
    best_params: Dict[str, float]
    min_sse: float
    best_params, min_sse = find_best_params(t_actual, P_actual, K_range, gamma_range)

    print("最適パラメータが見つかりました:")
    print(f"  γ (成長率): {best_params['gamma']:.4f}")
    print(f"  K (環境収容力): {best_params['K']}")
    print(f"  最小残差平方和 (SSE): {min_sse:.2f}")

    # 結果のプロットと保存
    plot_fit_result(t_actual, P_actual, best_params, args.output)

if __name__ == "__main__":
    main()
