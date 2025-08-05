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
    import time
    """
    ロジスティック方程式分析パイプラインを実行するメインスクリプト
    """
    # 設定の初期化
    model_params = ModelParameters(
        k_min=2000000000.0, # 20億
        k_max=3000000000.0, # 30億
        k_step=1000000.0, # 100万
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
    print(f"開始時間: {prediction_settings.start_year}")
    print(f"予測終了期間: {prediction_settings.forecast_end_t}")
    print(f"K範囲: {model_params.k_range.min_val}-{model_params.k_range.max_val} (step: {model_params.k_range.step})")
    print(f"γ範囲: {model_params.gamma_range.min_val}-{model_params.gamma_range.max_val} (step: {model_params.gamma_range.step})")
    print(f"パラメータ探索組み合わせ数: {search_info['total_combinations']:,}")
    
    # 必要なディレクトリの作成
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # データ抽出
    with yaspin(Spinners.line, text="Excelデータ抽出中") as spinner:
        try:
            time_array, value_array = DataExtractor.extract_from_directory(config.INPUT_DIR)
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"抽出失敗: {e}"
            return
    print("\n✅ Excelデータ抽出が完了しました")
    print(f"抽出データ点数: {len(time_array)}点")
    if len(time_array) == 0 or len(value_array) == 0:
        print("抽出されたデータが空です。処理を終了します。")
        return

    # パラメータフィッティング（設定とデータを注入）
    fitter = ParameterFitter(model_params, time_array, value_array)
    with yaspin(Spinners.line, text="最適なパラメータを探索中") as spinner:
        print("\n※最適なパラメータ探索には少し時間がかかる場合があります。しばらくお待ちください。")
        start_time = time.time()
        try:
            best_params, min_sse = fitter.fit_parameters()
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"パラメータ探索失敗: {e}"
            return
    
    print("\n✅ パラメータ探索が完了しました")
    elapsed = time.time() - start_time
    final_gamma = best_params['gamma']
    final_K = best_params['K']
    print(f"探索結果: γ={final_gamma:.4f}, K={final_K}, SSE={min_sse:.2f}")
    print(f"探索時間: {elapsed:.2f}秒")

    # フィッティング結果の可視化
    fitting_visualizer = FittingVisualizer(prediction_settings)
    with yaspin(Spinners.line, text="適合結果プロット中") as spinner:
        try:
            fitting_visualizer.plot_with_equation(
                time_array, value_array, fitter.get_fitted_equation(), config.FIT_RESULT_PNG
            )
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"プロット失敗: {e}"
            return
    
    print("\n✅ 適合結果プロットが完了しました")
    print(f"適合結果画像: {config.FIT_RESULT_PNG}")

    # 将来予測と可視化
    predictor = FuturePredictor(fitter.get_fitted_equation(), prediction_settings)
    forecast_visualizer = ForecastVisualizer(prediction_settings)
    with yaspin(Spinners.line, text="将来予測をプロット中") as spinner:
        try:
            forecast_time_array, forecast_value_array = predictor.predict(time_array, value_array)
            forecast_visualizer.plot_forecast(
                time_array, value_array, forecast_time_array, forecast_value_array, 
                config.FORECAST_RESULT_PNG
            )
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"予測・プロット失敗: {e}"
            return
    
    print("\n✅ 将来予測プロットが完了しました")
    print(f"フィッティングの結果の画像: {config.FIT_RESULT_PNG}")
    print(f"将来予測画像: {config.FORECAST_RESULT_PNG}")

    print(f"\n✅ パイプラインが正常に完了しました！")
    print(f"生成されたファイルは '{config.OUTPUT_DIR}' ディレクトリを確認してください。")

if __name__ == '__main__':
    main()
