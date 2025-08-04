import os
from typing import NoReturn
import numpy as np
import pandas as pd
import config.config as config
from yaspin import yaspin
from yaspin.spinners import Spinners

from scripts.data_conversion.excel_to_md import excel_to_markdown
from scripts.analysis.logistic_fitting import find_best_params
from scripts.analysis.logistic_prediction import predict_future
from scripts.visualization.plot_fitting import plot_fit_result
from scripts.visualization.plot_forecast import plot_forecast_result

def main() -> None:
    """
    ロジスティックモデル分析パイプラインを実行するメインスクリプト
    """
    # 必要なディレクトリの作成
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    with yaspin(Spinners.line, text="ExcelをMarkdownに変換中") as spinner:
        try:
            t_actual, P_actual, INPUT_EXCEL = excel_to_markdown(config.INPUT_DIR, config.OUTPUT_MD)
            spinner.ok("✅ ")
            spinner.text = "Excel→Markdown変換＆データ取得 完了"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"変換失敗: {e}"
            return

    with yaspin(Spinners.line, text="最適なパラメータを探索中") as spinner:
        try:
            best_params, min_sse = find_best_params(t_actual, P_actual, config.K_RANGE, config.GAMMA_RANGE)
            final_gamma = best_params['gamma']
            final_K = best_params['K']
            spinner.ok("✅ ")
            spinner.text = f"探索結果: γ={final_gamma:.4f}, K={final_K}, SSE={min_sse:.2f}"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"パラメータ探索失敗: {e}"
            return

    with yaspin(Spinners.line, text="適合結果プロット中") as spinner:
        try:
            plot_fit_result(t_actual, P_actual, best_params, config.FIT_RESULT_PNG)
            spinner.ok("✅ ")
            spinner.text = "適合結果プロット 完了"
        except Exception as e:
            spinner.fail("💥 ")
            spinner.text = f"プロット失敗: {e}"
            return

    with yaspin(Spinners.line, text="将来予測プロット中") as spinner:
        try:
            t_forecast, P_forecast = predict_future(t_actual, P_actual, final_gamma, final_K, config.FORECAST_END_T)
            plot_forecast_result(t_actual, P_actual, t_forecast, P_forecast, config.FORECAST_RESULT_PNG, start_year=config.START_YEAR)
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
