"""
将来予測機能
"""
from typing import Tuple
import numpy as np
from .logistic_equation import LogisticEquation


class FuturePredictor:
    """
    ロジスティック方程式による将来予測を行うクラス
    """
    
    def __init__(self, equation: LogisticEquation = None):
        """
        FuturePredictor の初期化
        
        Args:
            equation (LogisticEquation, optional): フィッティング済みの方程式
        """
        self.equation = equation if equation is not None else LogisticEquation()
    
    def set_equation(self, equation: LogisticEquation) -> None:
        """
        予測に使用する方程式を設定
        
        Args:
            equation (LogisticEquation): フィッティング済みの方程式
        """
        self.equation = equation
    
    def set_parameters(self, gamma: float, K: float) -> None:
        """
        方程式のパラメータを直接設定
        
        Args:
            gamma (float): 成長率
            K (float): 環境収容力
        """
        self.equation.set_parameters(gamma, K)
    
    def predict(
        self, 
        time_array: np.ndarray, 
        value_array: np.ndarray, 
        forecast_end_t: float,
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        フィッティングされたパラメータを使用して将来予測を行う
        
        Args:
            time_array (np.ndarray): 実績データの時刻
            value_array (np.ndarray): 実績データの値
            forecast_end_t (float): 予測の終了時刻
            dt (float): 時間刻み幅
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (予測時刻配列, 予測値配列)
        
        Raises:
            ValueError: 方程式のパラメータが設定されていない場合
        """
        if self.equation.gamma is None or self.equation.K is None:
            raise ValueError("方程式のパラメータが設定されていません。先にset_parameters()またはset_equation()を呼び出してください。")
        
        P0: float = value_array[0]
        t_start: float = time_array[0]
        
        # より細かい時間刻みで予測を実行
        t_forecast, P_forecast = self.equation.solve_runge_kutta(
            P0, t_start, forecast_end_t, dt
        )
        
        return t_forecast, P_forecast
    
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
            raise ValueError("方程式のパラメータが設定されていません。先にset_parameters()またはset_equation()を呼び出してください。")
        
        t_forecast, P_forecast = self.equation.solve_runge_kutta(
            last_value, last_time, forecast_end_t, dt
        )
        
        return t_forecast, P_forecast
