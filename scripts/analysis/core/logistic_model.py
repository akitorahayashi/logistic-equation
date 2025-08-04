"""
ロジスティックモデルの数式定義
"""

def logistic_model(P, gamma, K):
    """
    ロジスティックモデル微分方程式
    
    Args:
        P (float): 現在の人口値
        gamma (float): 成長率パラメータ
        K (float): 環境収容力（最大人口）
    
    Returns:
        float: 人口の変化率 dP/dt
    """
    return gamma * P * (1 - P / K)
