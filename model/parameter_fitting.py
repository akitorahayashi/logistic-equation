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
    
    def __init__(self):
        """
        ParameterFitter の初期化
        """
        self.equation = LogisticEquation()
        self.time_data: np.ndarray = None
        self.value_data: np.ndarray = None
        self.best_params: Dict[str, float] = None
        self.min_sse: float = None
    
    def load_data_from_excel(self, xlsx_path: str, header: int = 0) -> None:
        """
        xlsxファイルからtime, value列を読み込む
        
        Args:
            xlsx_path (str): Excelファイルのパス
            header (int): ヘッダー行の番号
        
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: 必要な列が存在しない場合
        """
        try:
            df = pd.read_excel(xlsx_path, header=header)
            if "time" in df.columns and "value" in df.columns:
                self.time_data = df["time"].values
                self.value_data = df["value"].values
            else:
                raise ValueError("Excelファイルに 'time' および 'value' 列が必要です。")
        except FileNotFoundError:
            raise FileNotFoundError(f"'{xlsx_path}' が見つかりません。")
        except Exception as e:
            raise Exception(f"Excelファイルの読み込みエラー: {e}")
    
    def set_data(self, time_array: np.ndarray, value_array: np.ndarray) -> None:
        """
        データを直接設定
        
        Args:
            time_array (np.ndarray): 時刻データ
            value_array (np.ndarray): 値データ
        """
        self.time_data = time_array
        self.value_data = value_array
    
    def fit_parameters(
        self, 
        K_range: np.ndarray, 
        gamma_range: np.ndarray
    ) -> Tuple[Dict[str, float], float]:
        """
        残差平方和(SSE)を最小化してパラメータをフィッティング
        
        Args:
            K_range (np.ndarray): 環境収容力Kの探索範囲
            gamma_range (np.ndarray): 成長率γの探索範囲
        
        Returns:
            Tuple[Dict[str, float], float]: (最適パラメータ辞書, 最小SSE値)
        
        Raises:
            ValueError: データが設定されていない場合
        """
        if self.time_data is None or self.value_data is None:
            raise ValueError("データが設定されていません。先にload_data_from_excel()またはset_data()を呼び出してください。")
        
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
