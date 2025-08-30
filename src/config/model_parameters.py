"""
ロジスティック方程式モデルのパラメータ設定
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class ParameterRange:
    """パラメータの範囲設定"""
    min_val: float
    max_val: float
    step: float
    
    def get_range(self) -> np.ndarray:
        """パラメータの探索範囲を生成"""
        return np.arange(self.min_val, self.max_val, self.step)
    
    def get_count(self) -> int:
        """探索ポイント数を取得"""
        return len(self.get_range())

class ModelParameters:
    """モデルパラメータを管理するクラス"""
    
    def __init__(self, 
                 k_min: float, 
                 k_max: float, 
                 k_step: float,
                 gamma_min: float, 
                 gamma_max: float, 
                 gamma_step: float):
        """
        初期化
        
        Args:
            k_min: Kパラメータの最小値
            k_max: Kパラメータの最大値
            k_step: Kパラメータの刻み幅
            gamma_min: γパラメータの最小値
            gamma_max: γパラメータの最大値
            gamma_step: γパラメータの刻み幅
        """
        self.k_range = ParameterRange(k_min, k_max, k_step)
        self.gamma_range = ParameterRange(gamma_min, gamma_max, gamma_step)
    
    def get_k_range(self) -> np.ndarray:
        """Kパラメータの探索範囲を取得"""
        return self.k_range.get_range()
    
    def get_gamma_range(self) -> np.ndarray:
        """γパラメータの探索範囲を取得"""
        return self.gamma_range.get_range()
    
    def get_search_info(self) -> dict:
        """探索情報を取得"""
        return {
            "k_count": self.k_range.get_count(),
            "gamma_count": self.gamma_range.get_count(),
            "total_combinations": self.k_range.get_count() * self.gamma_range.get_count()
        }
    
    def update_k_range(self, min_val: float, max_val: float, step: float) -> None:
        """Kパラメータの範囲を更新"""
        self.k_range = ParameterRange(min_val, max_val, step)
    
    def update_gamma_range(self, min_val: float, max_val: float, step: float) -> None:
        """γパラメータの範囲を更新"""
        self.gamma_range = ParameterRange(min_val, max_val, step)
