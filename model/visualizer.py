"""
可視化機能
"""
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from .logistic_equation import LogisticEquation

# 日本語フォントの設定 (macOS標準のヒラギノ角ゴシック)
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け対策


class FittingVisualizer:
    """
    パラメータフィッティング結果を可視化するクラス
    """
    
    def __init__(self):
        """
        FittingVisualizer の初期化
        """
        self.figure_size = (10, 6)
        self.dpi = 100
    
    def set_figure_properties(self, figure_size: tuple = (10, 6), dpi: int = 100) -> None:
        """
        図のプロパティを設定
        
        Args:
            figure_size (tuple): 図のサイズ (幅, 高さ)
            dpi (int): 解像度
        """
        self.figure_size = figure_size
        self.dpi = dpi
    
    def plot_with_equation(
        self, 
        t_actual: np.ndarray, 
        P_actual: np.ndarray, 
        equation: LogisticEquation, 
        output_path: str,
        title: str = "実データとロジスティック方程式の比較"
    ) -> None:
        """
        フィッティング済み方程式を使用してプロット
        
        Args:
            t_actual (np.ndarray): 実績データの時刻
            P_actual (np.ndarray): 実績データの人口値
            equation (LogisticEquation): フィッティング済みの方程式
            output_path (str): 出力画像のファイルパス
            title (str): グラフのタイトル
        
        Raises:
            ValueError: 方程式のパラメータが設定されていない場合
        """
        if equation.gamma is None or equation.K is None:
            raise ValueError("方程式のパラメータが設定されていません。")
        
        P0: float = P_actual[0]
        t_model, P_model = equation.solve_runge_kutta(P0, t_actual[0], t_actual[-1], 1.0)
        
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.plot(t_actual, P_actual, 'o', label='実データ')
        plt.plot(t_model, P_model, '-', 
                label=f'ロジスティック方程式 (γ={equation.gamma:.4f}, K={equation.K})')
        plt.title(title)
        plt.xlabel('時間')
        plt.ylabel('値')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        print(f"グラフを '{output_path}' として保存しました。")
        plt.close() # メモリリークを防ぐためにプロットを閉じる
    
    def plot_with_parameters(
        self, 
        t_actual: np.ndarray, 
        P_actual: np.ndarray, 
        best_params: Dict[str, float], 
        output_path: str,
        title: str = "実データとロジスティック方程式の比較"
    ) -> None:
        """
        パラメータ辞書を使用してプロット
        
        Args:
            t_actual (np.ndarray): 実績データの時刻
            P_actual (np.ndarray): 実績データの人口値
            best_params (Dict[str, float]): 最適化されたパラメータ {"gamma": float, "K": float}
            output_path (str): 出力画像のファイルパス
            title (str): グラフのタイトル
        """
        equation = LogisticEquation()
        equation.set_parameters(best_params["gamma"], best_params["K"])
        self.plot_with_equation(t_actual, P_actual, equation, output_path, title)


class ForecastVisualizer:
    """
    将来予測結果を可視化するクラス
    """
    
    def __init__(self):
        """
        ForecastVisualizer の初期化
        """
        self.figure_size = (12, 7)
        self.dpi = 100
    
    def set_figure_properties(self, figure_size: tuple = (12, 7), dpi: int = 100) -> None:
        """
        図のプロパティを設定
        
        Args:
            figure_size (tuple): 図のサイズ (幅, 高さ)
            dpi (int): 解像度
        """
        self.figure_size = figure_size
        self.dpi = dpi
    
    def plot_forecast(
        self, 
        t_actual: np.ndarray, 
        P_actual: np.ndarray, 
        t_forecast: np.ndarray, 
        P_forecast: np.ndarray, 
        output_path: str, 
        start_year: int = 1950,
        title: str = "ロジスティック方程式による人口推移と将来予測"
    ) -> None:
        """
        将来予測結果をプロット
        
        Args:
            t_actual (np.ndarray): 実績データの期間
            P_actual (np.ndarray): 実績データの値
            t_forecast (np.ndarray): 予測時刻の配列
            P_forecast (np.ndarray): 予測値の配列
            output_path (str): 出力画像のファイルパス
            start_year (int): プロットのx軸の開始年
            title (str): グラフのタイトル
        """
        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # 実績データのプロット (x軸を開始年で調整)
        actual_years: np.ndarray = t_actual + start_year
        plt.plot(actual_years, P_actual, 'o', 
                label=f'実際の人口データ ({actual_years[0]}-{actual_years[-1]}年)', 
                markersize=8, zorder=10)

        # 予測結果のプロット
        forecast_years: np.ndarray = t_forecast + start_year
        plt.plot(forecast_years, P_forecast, '-', 
                label=f'ロジスティック方程式による将来予測 ({int(forecast_years[-1])}年まで)')

        plt.title(title)
        plt.xlabel('年')
        plt.ylabel('人口（千人）')
        
        # 予測開始時点に垂直線を追加
        plt.axvline(x=actual_years[-1], color='gray', linestyle='--', 
                   label=f'予測開始 ({actual_years[-1]}年)')
        
        plt.legend()
        plt.grid(True)

        # グラフをファイルに保存
        plt.savefig(output_path)
        print(f"予測グラフを '{output_path}' として保存しました。")
        plt.close() # メモリリークを防ぐためにプロットを閉じる
