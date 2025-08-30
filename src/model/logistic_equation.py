"""
ロジスティック方程式の数学的定義と数値解法
"""
from typing import Tuple, Optional
import numpy as np


class LogisticEquation:
    """
    ロジスティック方程式のクラス
    
    dV/dt = gamma * V * (1 - V/K)
    
    Attributes:
        gamma (float): 成長率
        K (float): 環境収容力
    """
    
    def __init__(self, gamma: float, K: float):
        """
        ロジスティック方程式の初期化
        
        Args:
            gamma (float): 成長率
            K (float): 環境収容力
        """
        self.gamma = gamma
        self.K = K
    
    def differential_equation(self, t: float, v: float) -> float:
        """
        ロジスティック方程式: dV/dt = gamma * V * (1 - V/K)
        
        Args:
            t (float): 時間
            v (float): 現在の値
        
        Returns:
            float: 値の変化率 dV/dt
        
        Raises:
            ValueError: パラメータが設定されていない場合
        """
        if self.gamma is None or self.K is None:
            raise ValueError("パラメータ gamma と K を先に設定してください")
        
        # Kが0に近い場合のオーバーフローを防止
        if np.isclose(self.K, 0):
            return 0
            
        return self.gamma * v * (1 - v / self.K)
    
    def solve_runge_kutta(
        self, 
        v0: float,
        t_start: float, 
        t_end: float, 
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        4次ルンゲ・クッタ法による数値積分
        
        Args:
            v0 (float): 初期値
            t_start (float): 開始時刻
            t_end (float): 終了時刻
            dt (float): 時間刻み
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (時刻配列, 値の配列)
        
        Raises:
            ValueError: パラメータが設定されていない場合
        """
        if self.gamma is None or self.K is None:
            raise ValueError("パラメータ gamma と K を先に設定してください")
        
        n_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, n_steps)
        vs = np.zeros(n_steps)
        vs[0] = v0
        
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            for i in range(1, n_steps):
                try:
                    k1 = dt * self.differential_equation(t[i-1], vs[i-1])
                    k2 = dt * self.differential_equation(t[i-1] + dt/2, vs[i-1] + k1/2)
                    k3 = dt * self.differential_equation(t[i-1] + dt/2, vs[i-1] + k2/2)
                    k4 = dt * self.differential_equation(t[i-1] + dt, vs[i-1] + k3)
                    vs[i] = vs[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
                except FloatingPointError:
                    # 計算が発散した場合は、残りの値を無限大で埋める
                    vs[i:] = np.inf
                    break
        
        return t, vs
