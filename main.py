import os
from typing import NoReturn
import numpy as np
import pandas as pd
import config.config as config
from config.model_parameters import ModelParameters
from config.prediction_settings import PredictionSettings
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
    # 設定の初期化
    model_params = ModelParameters(
        k_min=2000000.0,
        k_max=3000000.0,
        k_step=10000.0,
        gamma_min=0.02,
        gamma_max=0.04,
        gamma_step=0.0005
    )
    prediction_settings = PredictionSettings(
        start_year=1950,
        forecast_end_t=250
    )
    
    # 設定情報の表示
    search_info = model_params.get_search_info()
    print(f"=== 設定情報 ===")
    print(f"開始年: {prediction_settings.start_year}")
    print(f"予測終了期間: {prediction_settings.forecast_end_t}")
    print(f"K範囲: {model_params.k_range.min_val}-{model_params.k_range.max_val} (step: {model_params.k_range.step})")
    print(f"γ範囲: {model_params.gamma_range.min_val}-{model_params.gamma_range.max_val} (step: {model_params.gamma_range.step})")
    print(f"パラメータ探索組み合わせ数: {search_info['total_combinations']:,}")
    print()
    
    # 必要なディレクトリの作成
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # データ抽出
    with yaspin(Spinners.line, text="Excelデータ抽出中") as spinner:
        try:
            time_array, value_array = DataExtractor.extract_from_directory(config.INPUT_DIR)
            
            # データの妥当性検証用にextractorインスタンスを作成
            xlsx_files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith('.xlsx')]
            excel_path = os.path.join(config.INPUT_DIR, xlsx_files[0])
            extractor = DataExtractor(excel_path)
            extractor.extract_data()  # データを内部に設定
            
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

    # パラメータフィッティング（設定とデータを注入）
    fitter = ParameterFitter(model_params, time_array, value_array)

    with yaspin(Spinners.line, text="最適なパラメータを探索中") as spinner:
        try:
            best_params, min_sse = fitter.fit_parameters()
            final_gamma = best_params['gamma']
            final_K = best_params['K']
            spinner.ok("✅ ")
            spinner.text = f"探索結果: γ={final_gamma:.4f}, K={final_K}, SSE={min_sse:.2f}"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"パラメータ探索失敗: {e}"
            return

    # フィッティング結果の可視化
    fitting_visualizer = FittingVisualizer(prediction_settings)
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
    predictor = FuturePredictor(fitter.get_fitted_equation(), prediction_settings)
    forecast_visualizer = ForecastVisualizer(prediction_settings)
    
    with yaspin(Spinners.line, text="将来予測プロット中") as spinner:
        try:
            forecast_time_array, forecast_value_array = predictor.predict(time_array, value_array)
            forecast_visualizer.plot_forecast(
                time_array, value_array, forecast_time_array, forecast_value_array, 
                config.FORECAST_RESULT_PNG
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
