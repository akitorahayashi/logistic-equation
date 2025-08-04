import os
import numpy as np
import pandas as pd
import config.config as config

from scripts.data_conversion.excel_to_md import excel_to_markdown
from scripts.analysis.logistic_fitting import find_best_params
from scripts.analysis.logistic_prediction import predict_future
from scripts.visualization.plot_fitting import plot_fit_result
from scripts.visualization.plot_forecast import plot_forecast_result

def main():
    """
    ロジスティックモデル分析パイプラインを実行するメインスクリプト
    """
    # 1. inputディレクトリのxlsxファイルを特定
    xlsx_files = [f for f in os.listdir(config.INPUT_DIR) if f.endswith('.xlsx')]
    if len(xlsx_files) != 1:
        print(f"エラー: inputディレクトリにxlsxファイルが1つだけ存在する必要があります（現在: {len(xlsx_files)}件）")
        return
    INPUT_EXCEL = os.path.join(config.INPUT_DIR, xlsx_files[0])
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- パイプライン実行 ---

    # 1. ExcelをMarkdownに変換
    print("--- 1. ExcelをMarkdownに変換中 ---")
    excel_to_markdown(INPUT_EXCEL, config.OUTPUT_MD)

    # 2. データの読み込みと準備
    print("\n--- 2. データの読み込みと準備 ---")
    try:
        df = pd.read_excel(INPUT_EXCEL, header=config.HEADER_ROW)
        t_actual = df[config.TIME_COL].values
        P_actual = df[config.VALUE_COL].values
        print(f"'{INPUT_EXCEL}' からデータを正常に読み込みました。")
    except Exception as e:
        print(f"エラー: データ読み込みに失敗しました。 {e}")
        return

    # 3. 最適なロジスティックモデルのパラメータを探索
    print("\n--- 3. 最適なパラメータを探索中 ---")
    best_params, min_sse = find_best_params(t_actual, P_actual, config.K_RANGE, config.GAMMA_RANGE)
    final_gamma = best_params['gamma']
    final_K = best_params['K']
    print(f"探索結果: γ = {final_gamma:.4f}, K = {final_K}")
    print(f"最小残差平方和 (SSE): {min_sse:.2f}")

    # 4. パラメータ適合結果をプロット
    print("\n--- 4. 適合結果をプロット中 ---")
    # `plot_fit_result` は汎用的なラベルを使用するため、このまま呼び出す
    plot_fit_result(t_actual, P_actual, best_params, config.FIT_RESULT_PNG)

    # 5. 将来の値を予測し、結果をプロット
    print("\n--- 5. 将来予測をプロット中 ---")
    t_forecast, P_forecast = predict_future(t_actual, P_actual, final_gamma, final_K, config.FORECAST_END_T)
    plot_forecast_result(t_actual, P_actual, t_forecast, P_forecast, config.FORECAST_RESULT_PNG, start_year=config.START_YEAR)

    print(f"\n--- パイプラインが正常に完了しました！ ---")
    print(f"生成されたファイルは '{config.OUTPUT_DIR}' ディレクトリを確認してください。")

if __name__ == '__main__':
    main()
