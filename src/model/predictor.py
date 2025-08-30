"""
将来予測機能
"""
import os
import pandas as pd
from typing import Tuple
import numpy as np
from .logistic_equation import LogisticEquation
from config.prediction_settings import PredictionSettings
import config.config as config


class FuturePredictor:
    """
    ロジスティック方程式による将来予測を行うクラス
    """
    
    def __init__(self, equation: LogisticEquation, prediction_settings: PredictionSettings):
        """
        FuturePredictor の初期化
        
        Args:
            equation (LogisticEquation): フィッティング済みの方程式
            prediction_settings: PredictionSettingsインスタンス
        """
        self.equation = equation
        self.prediction_settings = prediction_settings
    
    def predict(
        self, 
        time_array: np.ndarray, 
        value_array: np.ndarray, 
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        フィッティングされたパラメータを使用して将来予測を行う
        予測終了時刻は設定から取得する
        
        Args:
            time_array (np.ndarray): 実績データの時刻
            value_array (np.ndarray): 実績データの値
            dt (float): 時間刻み幅
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (予測時刻配列, 予測値配列)
        
        Raises:
            ValueError: 方程式のパラメータが設定されていない場合
        """
        if self.equation.gamma is None or self.equation.K is None:
            raise ValueError("方程式のパラメータが設定されていません。")
        
        # 設定から予測終了時刻を取得
        forecast_end_t = self.prediction_settings.forecast_end_t
        
        v0: float = value_array[0]
        t_start: float = time_array[0]
        
        # より細かい時間刻みで予測を実行
        t_forecast, v_forecast = self.equation.solve_runge_kutta(
            v0, t_start, forecast_end_t, dt
        )
        
        return t_forecast, v_forecast
    
    def predict_from_last_point(
        self, 
        last_time: float,
        last_value: float,
        forecast_end_t: float,
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        最後のデータポイントから将来予測を開始
        
        Args:
            last_time (float): 最後の時刻
            last_value (float): 最後の値
            forecast_end_t (float): 予測の終了時刻
            dt (float): 時間刻み幅
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (予測時刻配列, 予測値配列)
        """
        if self.equation.gamma is None or self.equation.K is None:
            raise ValueError("方程式のパラメータが設定されていません。")
        
        t_forecast, v_forecast = self.equation.solve_runge_kutta(
            last_value, last_time, forecast_end_t, dt
        )
        
        return t_forecast, v_forecast

    def save_prediction_to_excel(
        self,
        t_forecast: np.ndarray,
        v_forecast: np.ndarray,
        filename: str,
        interval: int = 1  # デフォルトは1年ごと
    ) -> str:
        """
        予測結果をExcelファイルに保存する

        Args:
            t_forecast (np.ndarray): 予測時間配列
            v_forecast (np.ndarray): 予測値配列
            filename (str): 出力ファイル名
            interval (int): データを抽出する時間間隔

        Returns:
            str: 保存したファイルのパス
        """
        # 時間が整数のインデックスを抽出
        indices = np.where(t_forecast % interval == 0)[0]
        
        # 重複を削除
        unique_indices = np.unique(indices)
        
        # データを抽出
        time_points = t_forecast[unique_indices]
        value_points = v_forecast[unique_indices]

        df = pd.DataFrame({
            'time': time_points,
            'value': value_points
        })

        output_path = os.path.join(config.OUTPUT_DIR, filename)
        df.to_excel(output_path, index=False, engine='openpyxl')
        return output_path
