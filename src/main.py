import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

# --- アプリケーションのコアロジックとUIコンポーネントのインポート ---
import os
from components.sidebar import render_sidebar
from components.results_display import render_results
from schemas.model_parameters_schema import ModelParametersSchema
from schemas.prediction_settings_schema import PredictionSettingsSchema
from services.parameter_fitting_service import ParameterFitterService
from services.predictor_service import FuturePredictorService
from services.visualizer_service import (
    FittingVisualizerService,
    ForecastVisualizerService,
)

st.set_page_config(page_title="ロジスティック方程式分析ツール", layout="wide")


def extract_data_from_uploaded_file(
    uploaded_file,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """アップロードされたExcelファイルからデータを抽出する"""
    try:
        df = pd.read_excel(uploaded_file, header=None)
        if df.shape[1] < 2:
            st.error("Excelファイルには少なくとも2つの列（時間と値）が必要です。")
            return None

        time_array = df.iloc[:, 0].values
        value_array = df.iloc[:, 1].values

        # ファイル名から拡張子を除いた部分を取得
        filename = os.path.splitext(uploaded_file.name)[0]

        return time_array, value_array, filename
    except Exception as e:
        st.error(f"Excelファイルの読み込み中にエラーが発生しました: {e}")
        return None


def run_analysis(settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    ユーザー設定に基づいて分析パイプラインを実行する。
    """
    uploaded_file = settings.get("uploaded_file")
    if not uploaded_file:
        st.warning("分析するファイルをアップロードしてください。")
        return None

    extraction_result = extract_data_from_uploaded_file(uploaded_file)
    if not extraction_result:
        return None

    time_array, value_array, excel_filename = extraction_result

    try:
        # 1. 設定の初期化
        model_params = ModelParametersSchema(
            k_min=settings["k_min"],
            k_max=settings["k_max"],
            k_step=settings["k_step"],
            gamma_min=settings["gamma_min"],
            gamma_max=settings["gamma_max"],
            gamma_step=settings["gamma_step"],
        )
        prediction_settings = PredictionSettingsSchema(
            start_year=settings["start_year"], forecast_end_t=settings["forecast_end_t"]
        )

        # 2. パラメータフィッティング
        fitter = ParameterFitterService(model_params, time_array, value_array)
        best_params, min_sse = fitter.fit_parameters()

        # 3. フィッティング結果の可視化
        fitting_visualizer = FittingVisualizerService(prediction_settings)
        fitting_fig = fitting_visualizer.plot_with_equation(
            time_array, value_array, fitter.get_fitted_equation()
        )

        # 4. 将来予測
        predictor = FuturePredictorService(
            fitter.get_fitted_equation(), prediction_settings
        )
        forecast_time_array, forecast_value_array = predictor.predict(
            time_array, value_array
        )

        # 5. 予測結果の可視化
        forecast_visualizer = ForecastVisualizerService(prediction_settings)
        forecast_fig = forecast_visualizer.plot_forecast(
            time_array, value_array, forecast_time_array, forecast_value_array
        )

        # 6. 予測データをDataFrameに保存
        forecast_df = predictor.create_prediction_dataframe(
            forecast_time_array, forecast_value_array
        )

        return {
            "best_params": best_params,
            "min_sse": min_sse,
            "fitting_fig": fitting_fig,
            "forecast_fig": forecast_fig,
            "forecast_df": forecast_df,
            "excel_filename": excel_filename,
        }

    except Exception as e:
        st.error(f"分析中にエラーが発生しました: {e}")
        # スタックトレースをログに出力するとデバッグに役立つ
        import traceback

        st.error(f"詳細: {traceback.format_exc()}")
        return None


def main():
    """
    Streamlitアプリケーションのメイン関数
    """
    # サイドバーをレンダリングし、ユーザー設定を取得
    user_settings = render_sidebar()

    # メインパネルの表示
    st.title("ロジスティック方程式による時系列データ分析")

    # セッション状態で分析結果を管理
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    # 分析実行ボタンが押されたら分析を実行
    if user_settings["run_analysis"]:
        with st.spinner(
            "分析を実行中...パラメータ探索には時間がかかる場合があります。"
        ):
            results = run_analysis(user_settings)
            st.session_state.analysis_results = results

    # 分析結果があれば表示
    if st.session_state.analysis_results:
        render_results(st.session_state.analysis_results)
    else:
        # 初期表示または分析前の状態
        st.info("左のサイドバーから設定を行い、「分析実行」ボタンを押してください。")
        st.markdown(
            """
        ### 使い方
        1.  **データファイルのアップロード**:
            - 分析したい時系列データを含むExcelファイル（.xlsx）をアップロードします。
            - Excelの1列目を時間（例：年）、2列目を観測値（例：人口）としてください。ヘッダーは不要です。
        2.  **パラメータ探索範囲の設定**:
            - **環境収容力 (K)** と **成長率 (γ)** の探索範囲とステップ（刻み幅）を指定します。
            - Kの値が大きい場合は、入力しやすいように単位（万、億、兆）を選択できます。
        3.  **予測期間の設定**:
            - データの開始年と、何年先まで予測したいかを設定します。
        4.  **分析の実行**:
            - 「分析実行」ボタンをクリックすると、パラメータ探索と将来予測が始まります。

        分析が完了すると、このエリアに結果（最適パラメータ、グラフ、予測データ）が表示されます。
        """
        )


if __name__ == "__main__":
    main()
