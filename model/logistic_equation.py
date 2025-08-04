"""
ロジスティック方程式の数学的定義と数値解法
"""
from typing import Tuple, Optional
import numpy as np


class LogisticEquation:
    """
    ロジスティック方程式のクラス
    
    dP/dt = gamma * P * (1 - P/K)
    
    Attributes:
        gamma (float): 成長率
        K (float): 環境収容力
    """
    
    def __init__(self, gamma: Optional[float] = None, K: Optional[float] = None):
        """
        ロジスティック方程式の初期化
        
        Args:
            gamma (Optional[float]): 成長率
            K (Optional[float]): 環境収容力
        """
        self.gamma = gamma
        self.K = K
    
    def set_parameters(self, gamma: float, K: float) -> None:
        """
        方程式のパラメータを設定
        
        Args:
            gamma (float): 成長率
            K (float): 環境収容力
        """
        self.gamma = gamma
        self.K = K
    
    def differential_equation(self, t: float, P: float) -> float:
        """
        ロジスティック方程式: dP/dt = gamma * P * (1 - P/K)
        
        Args:
            t (float): 時間
            P (float): 現在の人口
        
        Returns:
            float: 人口の変化率 dP/dt
        
        Raises:
            ValueError: パラメータが設定されていない場合
        """
        if self.gamma is None or self.K is None:
            raise ValueError("パラメータ gamma と K を先に設定してください")
        return self.gamma * P * (1 - P / self.K)
    
    def solve_runge_kutta(
        self, 
        P0: float, 
        t_start: float, 
        t_end: float, 
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        4次ルンゲ・クッタ法による数値積分
        
        Args:
            P0 (float): 初期値
            t_start (float): 開始時刻
            t_end (float): 終了時刻
            dt (float): 時間刻み
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (時刻配列, 人口配列)
        
        Raises:
            ValueError: パラメータが設定されていない場合
        """
        if self.gamma is None or self.K is None:
            raise ValueError("パラメータ gamma と K を先に設定してください")
        
        n_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, n_steps)
        P = np.zeros(n_steps)
        P[0] = P0
        
        for i in range(1, n_steps):
            k1 = dt * self.differential_equation(t[i-1], P[i-1])
            k2 = dt * self.differential_equation(t[i-1] + dt/2, P[i-1] + k1/2)
            k3 = dt * self.differential_equation(t[i-1] + dt/2, P[i-1] + k2/2)
            k4 = dt * self.differential_equation(t[i-1] + dt, P[i-1] + k3)
            P[i] = P[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return t, P
