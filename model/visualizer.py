"""
可視化機能
"""
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from .logistic_equation import LogisticEquation
from config.prediction_settings import PredictionSettings

# 日本語フォントの設定 (macOS標準のヒラギノ角ゴシック)
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け対策


class FittingVisualizer:
    """
    パラメータフィッティング結果を可視化するクラス
    """
    
    def __init__(self, prediction_settings: Optional[PredictionSettings] = None):
        """
        FittingVisualizer の初期化
        
        Args:
            prediction_settings: PredictionSettingsインスタンス（オプション）
        """
        self.prediction_settings = prediction_settings
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
        time_array: np.ndarray, 
        value_array: np.ndarray, 
        equation: LogisticEquation, 
        output_path: str,
        title: str = "実データとロジスティック方程式の比較"
    ) -> None:
        """
        フィッティング済み方程式を使用してプロット
        
        Args:
            time_array (np.ndarray): 実績データの時刻
            value_array (np.ndarray): 実績データの値
            equation (LogisticEquation): フィッティング済みの方程式
            output_path (str): 出力画像のファイルパス
            title (str): グラフのタイトル
        
        Raises:
            ValueError: 方程式のパラメータが設定されていない場合
        """
        if equation.gamma is None or equation.K is None:
            raise ValueError("方程式のパラメータが設定されていません。")
        
        v0: float = value_array[0]
        time_model, value_model = equation.solve_runge_kutta(v0, time_array[0], time_array[-1], 1.0)
        
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        plt.plot(time_array, value_array, 'o', label='実データ')
        plt.plot(time_model, value_model, '-', 
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
        time_array: np.ndarray, 
        value_array: np.ndarray, 
        best_params: Dict[str, float], 
        output_path: str,
        title: str = "実データとロジスティック方程式の比較"
    ) -> None:
        """
        パラメータ辞書を使用してプロット
        
        Args:
            time_array (np.ndarray): 実績データの時刻
            value_array (np.ndarray): 実績データの値
            best_params (Dict[str, float]): 最適化されたパラメータ {"gamma": float, "K": float}
            output_path (str): 出力画像のファイルパス
            title (str): グラフのタイトル
        """
        equation = LogisticEquation(best_params["gamma"], best_params["K"])
        self.plot_with_equation(time_array, value_array, equation, output_path, title)


class ForecastVisualizer:
    """
    将来予測結果を可視化するクラス
    """
    
    def __init__(self, prediction_settings: PredictionSettings):
        """
        ForecastVisualizer の初期化
        
        Args:
            prediction_settings: PredictionSettingsインスタンス（必須）
        """
        self.prediction_settings = prediction_settings
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
        time_array: np.ndarray, 
        value_array: np.ndarray, 
        forecast_time_array: np.ndarray, 
        forecast_value_array: np.ndarray, 
        output_path: str, 
        title: str = "ロジスティック方程式による時系列データと将来予測"
    ) -> None:
        """
        将来予測結果をプロット
        
        Args:
            time_array (np.ndarray): 実績データの期間
            value_array (np.ndarray): 実績データの値
            forecast_time_array (np.ndarray): 予測時刻の配列
            forecast_value_array (np.ndarray): 予測値の配列
            output_path (str): 出力画像のファイルパス
            title (str): グラフのタイトル
        """
        plt.figure(figsize=self.figure_size, dpi=self.dpi)

        # 実績データの表示用時間軸を計算（年固定）
        actual_display_time = time_array + self.prediction_settings.start_year
        forecast_display_time = forecast_time_array + self.prediction_settings.start_year

        # 実績データのプロット
        plt.plot(actual_display_time, value_array, 'o', 
                label=f'実際のデータ ({actual_display_time[0]:.0f}-{actual_display_time[-1]:.0f}年)', 
                markersize=8, zorder=10)

        # 予測結果のプロット
        plt.plot(forecast_display_time, forecast_value_array, '-', 
                label=f'ロジスティック方程式による将来予測 ({forecast_display_time[-1]:.0f}年まで)')

        plt.title(title)
        plt.xlabel('年')
        plt.ylabel('値')
        
        # 予測開始時点に垂直線を追加
        plt.axvline(x=actual_display_time[-1], color='gray', linestyle='--', 
                   label=f'予測開始 ({actual_display_time[-1]:.0f}年)')
        
        plt.legend()
        plt.grid(True)

        # グラフをファイルに保存
        plt.savefig(output_path)
        print(f"予測グラフを '{output_path}' として保存しました。")
        plt.close() # メモリリークを防ぐためにプロットを閉じる
