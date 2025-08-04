"""
ロジスティックモデルによる将来予測
"""
from .core.logistic_model import logistic_model
from .core.numerical_methods import runge_kutta_4th_order

def predict_future(t_actual, P_actual, gamma, K, forecast_end_t):
    """
    ロジスティックモデルを使用して将来予測を行う
    
    Args:
        t_actual (np.array): 実績データの期間
        P_actual (np.array): 実績データの値
        gamma (float): 成長率パラメータ
        K (float): 環境収容力
        forecast_end_t (int): 予測の最終期間
    
    Returns:
        tuple: (予測時刻の配列, 予測値の配列)
    """
    P0 = P_actual[0]
    t_start_forecast = t_actual[0]
    dt_forecast = 1.0

    # 将来予測の計算
    t_forecast, P_forecast = runge_kutta_4th_order(
        P0, t_start_forecast, forecast_end_t, dt_forecast, gamma, K, logistic_model
    )
    
    return t_forecast, P_forecast
