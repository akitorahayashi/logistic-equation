"""
ロジスティック方程式の予測設定
"""

class PredictionSettings:
    """予測設定を管理するクラス（年単位固定）"""
    
    def __init__(self, start_year: int, forecast_end_t: int):
        """
        初期化
        
        Args:
            start_year: 実績データの開始年
            forecast_end_t: 予測の最終期間（年単位）
        """
        self.start_year = start_year
        self.forecast_end_t = forecast_end_t
    
    def get_time_unit_label(self) -> str:
        """時間単位のラベルを取得（年固定）"""
        return "年"
