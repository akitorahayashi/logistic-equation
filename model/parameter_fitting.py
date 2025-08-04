"""
パラメータフィッティング機能
"""
import sys
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from .logistic_equation import LogisticEquation


class ParameterFitter:
    """
    ロジスティック方程式のパラメータフィッティングを行うクラス
    """
    
    def __init__(self, model_params, time_data: np.ndarray, value_data: np.ndarray):
        """
        ParameterFitter の初期化
        
        Args:
            model_params: ModelParametersインスタンス
            time_data: 時刻データ
            value_data: 値データ
        """
        self.equation = LogisticEquation(0.01, 1000000)  # ダミー値で初期化
        self.model_params = model_params
        self.time_data = time_data
        self.value_data = value_data
        self.best_params: Dict[str, float] = None
        self.min_sse: float = None

    def fit_parameters(self) -> Tuple[Dict[str, float], float]:
        """
        残差平方和(SSE)を最小化してパラメータをフィッティング
        設定から探索範囲を取得して使用する
        
        Returns:
            Tuple[Dict[str, float], float]: (最適パラメータ辞書, 最小SSE値)
        
        Raises:
            ValueError: データが設定されていない場合
        """
        if self.time_data is None or self.value_data is None:
            raise ValueError("データが設定されていません。")
        
        # 設定から探索範囲を取得
        K_range = self.model_params.get_k_range()
        gamma_range = self.model_params.get_gamma_range()
        
        P0: float = self.value_data[0]
        best_params: Dict[str, float] = {"gamma": 0.0, "K": 0.0}
        min_sse: float = np.inf

        for K in K_range:
            for gamma in gamma_range:
                # 一時的にパラメータを設定
                self.equation.set_parameters(gamma, K)
                t_model, P_model = self.equation.solve_runge_kutta(
                    P0, self.time_data[0], self.time_data[-1], 1.0
                )
                
                # SSE計算のために、実績データ点に対応するモデル上の値を内挿
                model_P_at_actual_t: np.ndarray = np.interp(self.time_data, t_model, P_model)
                sse: float = np.sum((self.value_data - model_P_at_actual_t)**2)
                
                if sse < min_sse:
                    min_sse = sse
                    best_params["gamma"] = gamma
                    best_params["K"] = K
        
        self.best_params = best_params
        self.min_sse = min_sse
        # 最適パラメータを方程式に設定
        self.equation.set_parameters(best_params["gamma"], best_params["K"])
        
        return best_params, min_sse
    
    def get_best_params(self) -> Dict[str, float]:
        """
        最適化されたパラメータを取得
        
        Returns:
            Dict[str, float]: 最適パラメータ辞書
        
        Raises:
            ValueError: フィッティングが実行されていない場合
        """
        if self.best_params is None:
            raise ValueError("パラメータフィッティングが実行されていません。先にfit_parameters()を呼び出してください。")
        return self.best_params
    
    def get_fitted_equation(self) -> LogisticEquation:
        """
        フィッティングされた方程式を取得
        
        Returns:
            LogisticEquation: パラメータがフィッティングされた方程式
        """
        return self.equation
