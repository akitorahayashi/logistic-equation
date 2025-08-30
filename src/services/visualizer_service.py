"""
可視化機能
"""

from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from models.logistic_equation import LogisticEquationModel
from schemas.prediction_settings_schema import PredictionSettingsSchema

# 日本語フォントの設定 (macOS標準のヒラギノ角ゴシック)
# Streamlit Cloud/Linux環境を考慮し、フォントが見つからない場合はスキップ
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策
except Exception:
    pass


def _get_scaled_data_and_unit(value_array: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    データ配列の最大値に応じて、適切な単位とスケーリングされたデータ配列を返す。
    単位は「万」「億」「兆」をサポート。
    """
    max_val = np.max(value_array)
    if max_val >= 10**12:
        return value_array / 10**12, "兆"
    elif max_val >= 10**8:
        return value_array / 10**8, "億"
    elif max_val >= 10**4:
        return value_array / 10**4, "万"
    else:
        return value_array, ""


class FittingVisualizerService:
    """
    パラメータフィッティング結果を可視化するサービスクラス
    """

    def __init__(self, prediction_settings: PredictionSettingsSchema):
        """
        FittingVisualizerService の初期化

        Args:
            prediction_settings: PredictionSettingsSchemaインスタンス
        """
        self.prediction_settings = prediction_settings
        self.figure_size = (10, 6)
        self.dpi = 100

    def set_figure_properties(
        self, figure_size: tuple = (10, 6), dpi: int = 100
    ) -> None:
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
        equation: LogisticEquationModel,
        title: str = "実データとロジスティック方程式の比較",
    ) -> matplotlib.figure.Figure:
        """
        フィッティング済み方程式を使用してプロットし、Figureオブジェクトを返す

        Args:
            time_array (np.ndarray): 実績データの時刻
            value_array (np.ndarray): 実績データの値
            equation (LogisticEquationModel): フィッティング済みの方程式
            title (str): グラフのタイトル

        Returns:
            matplotlib.figure.Figure: プロットされたグラフのFigureオブジェクト

        Raises:
            ValueError: 方程式のパラメータが設定されていない場合
        """
        if equation.gamma is None or equation.K is None:
            raise ValueError("方程式のパラメータが設定されていません。")

        v0: float = value_array[0]
        time_model, value_model = equation.solve_runge_kutta(
            v0, time_array[0], time_array[-1], 0.1
        )

        display_time_array = time_array + self.prediction_settings.start_year
        display_time_model = time_model + self.prediction_settings.start_year

        all_values_for_scaling = np.concatenate(
            [value_array, value_model, [equation.K]]
        )
        _, unit = _get_scaled_data_and_unit(all_values_for_scaling)

        scale = 1
        if unit == "兆":
            scale = 10**12
        elif unit == "億":
            scale = 10**8
        elif unit == "万":
            scale = 10**4

        scaled_value_array = value_array / scale
        scaled_value_model = value_model / scale
        scaled_K = equation.K / scale

        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(display_time_array, scaled_value_array, "o", label="実データ")
        ax.plot(
            display_time_model,
            scaled_value_model,
            "-",
            label=f"ロジスティック方程式 (γ={equation.gamma:.4f}, K={scaled_K:.2f}{unit})",
        )
        ax.set_title(title)
        ax.set_xlabel("時間")
        ax.set_ylabel(f"値 ({unit})")
        ax.legend()
        ax.grid(True)

        return fig

    def plot_with_parameters(
        self,
        time_array: np.ndarray,
        value_array: np.ndarray,
        best_params: Dict[str, float],
        title: str = "実データとロジスティック方程式の比較",
    ) -> matplotlib.figure.Figure:
        """
        パラメータ辞書を使用してプロットし、Figureオブジェクトを返す

        Args:
            time_array (np.ndarray): 実績データの時刻
            value_array (np.ndarray): 実績データの値
            best_params (Dict[str, float]): 最適化されたパラメータ {"gamma": float, "K": float}
            title (str): グラフのタイトル

        Returns:
            matplotlib.figure.Figure: プロットされたグラフのFigureオブジェクト
        """
        equation = LogisticEquationModel(best_params["gamma"], best_params["K"])
        return self.plot_with_equation(time_array, value_array, equation, title)


class ForecastVisualizerService:
    """
    将来予測結果を可視化するサービスクラス
    """

    def __init__(self, prediction_settings: PredictionSettingsSchema):
        """
        ForecastVisualizerService の初期化

        Args:
            prediction_settings: PredictionSettingsSchemaインスタンス（必須）
        """
        self.prediction_settings = prediction_settings
        self.figure_size = (12, 7)
        self.dpi = 100

    def set_figure_properties(
        self, figure_size: tuple = (12, 7), dpi: int = 100
    ) -> None:
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
        title: str = "ロジスティック方程式による時系列データと将来予測",
    ) -> matplotlib.figure.Figure:
        """
        将来予測結果をプロットし、Figureオブジェクトを返す

        Args:
            time_array (np.ndarray): 実績データの期間
            value_array (np.ndarray): 実績データの値
            forecast_time_array (np.ndarray): 予測時刻の配列
            forecast_value_array (np.ndarray): 予測値の配列
            title (str): グラフのタイトル

        Returns:
            matplotlib.figure.Figure: プロットされたグラフのFigureオブジェクト
        """
        fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        ax = fig.add_subplot(1, 1, 1)

        actual_display_time = time_array + self.prediction_settings.start_year
        forecast_display_time = (
            forecast_time_array + self.prediction_settings.start_year
        )

        all_values = np.concatenate([value_array, forecast_value_array])
        scaled_all_values, unit = _get_scaled_data_and_unit(all_values)
        scaled_value_array = scaled_all_values[: len(value_array)]
        scaled_forecast_value_array = scaled_all_values[len(value_array) :]

        ax.plot(
            actual_display_time,
            scaled_value_array,
            "o",
            label=f"実績データ ({actual_display_time[0]:.0f}-{actual_display_time[-1]:.0f})",
            markersize=8,
            zorder=10,
        )

        ax.plot(
            forecast_display_time,
            scaled_forecast_value_array,
            "-",
            label=f"ロジスティック方程式による将来予測 ({forecast_display_time[-1]:.0f}まで)",
        )

        ax.set_title(title)
        ax.set_xlabel("時間")
        ax.set_ylabel(f"値 ({unit})")

        ax.axvline(
            x=actual_display_time[-1],
            color="gray",
            linestyle="--",
            label=f"予測開始 ({actual_display_time[-1]:.0f})",
        )

        ax.legend()
        ax.grid(True)

        return fig
