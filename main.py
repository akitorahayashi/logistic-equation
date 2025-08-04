import os
from typing import NoReturn
import numpy as np
import pandas as pd
import config.config as config
from yaspin import yaspin
from yaspin.spinners import Spinners

from model import (
    DataExtractor, 
    ParameterFitter, 
    FuturePredictor, 
    FittingVisualizer, 
    ForecastVisualizer
)

def main() -> None:
    """
    ロジスティック方程式分析パイプラインを実行するメインスクリプト
    """
    # 必要なディレクトリの作成
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # データ抽出
    extractor = DataExtractor()
    with yaspin(Spinners.line, text="Excelデータ抽出中") as spinner:
        try:
            time_array, value_array = extractor.extract_from_directory(config.INPUT_DIR)
            
            # データの妥当性検証
            if not extractor.validate_data():
                raise ValueError("抽出されたデータに問題があります")
            
            # データ情報の表示
            data_info = extractor.get_data_info()
            spinner.ok("✅ ")
            spinner.text = f"データ抽出完了 ({data_info['data_points']}点)"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"抽出失敗: {e}"
            return

    # パラメータフィッティング
    fitter = ParameterFitter()
    fitter.set_data(time_array, value_array)

    with yaspin(Spinners.line, text="最適なパラメータを探索中") as spinner:
        try:
            best_params, min_sse = fitter.fit_parameters(config.K_RANGE, config.GAMMA_RANGE)
            final_gamma = best_params['gamma']
            final_K = best_params['K']
            spinner.ok("✅ ")
            spinner.text = f"探索結果: γ={final_gamma:.4f}, K={final_K}, SSE={min_sse:.2f}"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"パラメータ探索失敗: {e}"
            return

    # フィッティング結果の可視化
    fitting_visualizer = FittingVisualizer()
    with yaspin(Spinners.line, text="適合結果プロット中") as spinner:
        try:
            fitting_visualizer.plot_with_equation(
                time_array, value_array, fitter.get_fitted_equation(), config.FIT_RESULT_PNG
            )
            spinner.ok("✅ ")
            spinner.text = "適合結果プロット 完了"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"プロット失敗: {e}"
            return

    # 将来予測と可視化
    predictor = FuturePredictor(fitter.get_fitted_equation())
    forecast_visualizer = ForecastVisualizer()
    
    with yaspin(Spinners.line, text="将来予測プロット中") as spinner:
        try:
            t_forecast, value_forecast = predictor.predict(
                time_array, value_array, config.FORECAST_END_T
            )
            forecast_visualizer.plot_forecast(
                time_array, value_array, t_forecast, value_forecast, 
                config.FORECAST_RESULT_PNG, start_year=config.START_YEAR
            )
            spinner.ok("✅ ")
            spinner.text = "将来予測プロット 完了"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"予測・プロット失敗: {e}"
            return

    print(f"\n--- パイプラインが正常に完了しました！ ---")
    print(f"生成されたファイルは '{config.OUTPUT_DIR}' ディレクトリを確認してください。")

if __name__ == '__main__':
    main()
