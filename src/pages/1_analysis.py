import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import os
from io import BytesIO

# --- アプリケーションのコアロジックのインポート ---
# パスの問題を解決するために、プロジェクトルートをsys.pathに追加
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.model_parameters_schema import ModelParametersSchema
from schemas.prediction_settings_schema import PredictionSettingsSchema
from services.parameter_fitting_service import ParameterFitterService
from services.predictor_service import FuturePredictorService
from services.visualizer_service import (
    FittingVisualizerService,
    ForecastVisualizerService,
)

# --- UIコンポーネントの定義 (旧sidebar.pyとresults_display.pyの内容を統合) ---


def render_sidebar() -> Dict[str, Any]:
    """
    分析ページの設定サイドバーをレンダリングし、ユーザーの入力を収集する。
    """
    st.sidebar.title("分析設定")

    settings: Dict[str, Any] = {}

    # --- データアップロード ---
    st.sidebar.header("1. データファイルのアップロード")
    uploaded_file: Optional[BytesIO] = st.sidebar.file_uploader(
        "分析するExcelファイル (.xlsx) を選択してください", type=["xlsx"]
    )
    settings["uploaded_file"] = uploaded_file

    # --- パラメータ設定 ---
    st.sidebar.header("2. パラメータ探索範囲の設定")

    # 環境収容力 (K)
    st.sidebar.subheader("環境収容力 (K) の範囲")
    k_col1, k_col2 = st.sidebar.columns(2)
    with k_col1:
        k_unit = st.selectbox("Kの単位", ["", "万", "億", "兆"], index=2, key="k_unit")

    unit_multiplier = {"": 1, "万": 10**4, "億": 10**8, "兆": 10**12}
    multiplier = unit_multiplier[k_unit]

    k_min = st.sidebar.number_input(
        "最小値 (K_min)", value=20.0, min_value=0.0, step=1.0, format="%.2f"
    )
    k_max = st.sidebar.number_input(
        "最大値 (K_max)", value=30.0, min_value=0.0, step=1.0, format="%.2f"
    )
    k_step = st.sidebar.number_input(
        "ステップ (K_step)", value=5.0, min_value=0.01, step=1.0, format="%.2f"
    )

    settings["k_min"] = k_min * multiplier
    settings["k_max"] = k_max * multiplier
    settings["k_step"] = k_step * multiplier
    st.sidebar.caption(f"探索範囲: {settings['k_min']:,} 〜 {settings['k_max']:,}")

    # 成長率 (γ)
    st.sidebar.subheader("成長率 (γ) の範囲")
    gamma_min = st.sidebar.number_input(
        "最小値 (γ_min)", value=0.0285, min_value=0.0, step=0.0001, format="%.4f"
    )
    gamma_max = st.sidebar.number_input(
        "最大値 (γ_max)", value=0.0350, min_value=0.0, step=0.0001, format="%.4f"
    )
    gamma_step = st.sidebar.number_input(
        "ステップ (γ_step)", value=0.0005, min_value=0.0001, step=0.0001, format="%.4f"
    )

    settings["gamma_min"] = gamma_min
    settings["gamma_max"] = gamma_max
    settings["gamma_step"] = gamma_step

    # --- 予測期間設定 ---
    st.sidebar.header("3. 予測期間の設定")
    start_year = st.sidebar.number_input(
        "データ開始年", value=1950, min_value=0, step=1
    )
    forecast_end_t = st.sidebar.number_input(
        "予測年数 (データ開始年からの年数)", value=150, min_value=1, step=10
    )

    settings["start_year"] = start_year
    settings["forecast_end_t"] = forecast_end_t
    st.sidebar.caption(f"予測は {start_year + forecast_end_t} 年まで行われます。")

    # --- 実行ボタン ---
    st.sidebar.header("4. 分析の実行")
    run_analysis_button = st.sidebar.button(
        "分析実行", disabled=(uploaded_file is None)
    )
    settings["run_analysis"] = run_analysis_button

    if uploaded_file is None:
        st.sidebar.warning("分析を開始するには、ファイルをアップロードしてください。")

    return settings


def to_excel(df: pd.DataFrame) -> bytes:
    """DataFrameをインメモリのExcelファイルに変換する"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prediction")
    return output.getvalue()


def render_results(results: Dict[str, Any]) -> None:
    """分析結果をレンダリングする"""
    st.header("📈 分析結果")

    tab1, tab2, tab3 = st.tabs(["フィッティング結果", "将来予測", "予測データ"])

    with tab1:
        st.subheader("最適なパラメータ")
        best_params = results.get("best_params", {})
        min_sse = results.get("min_sse", 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("成長率 (γ)", f"{best_params.get('gamma', 0):.4f}")

        k_value = best_params.get("K", 0)
        if k_value >= 10**12:
            k_display = f"{k_value / 10**12:.2f} 兆"
        elif k_value >= 10**8:
            k_display = f"{k_value / 10**8:.2f} 億"
        elif k_value >= 10**4:
            k_display = f"{k_value / 10**4:.2f} 万"
        else:
            k_display = f"{k_value:.2f}"
        col2.metric("環境収容力 (K)", k_display)

        col3.metric("最小二乗誤差 (SSE)", f"{min_sse:,.2f}")

        st.subheader("適合結果のプロット")
        fitting_fig = results.get("fitting_fig")
        if fitting_fig:
            st.pyplot(fitting_fig)
        else:
            st.warning("適合プロットを生成できませんでした。")

    with tab2:
        st.subheader("将来予測のプロット")
        forecast_fig = results.get("forecast_fig")
        if forecast_fig:
            st.pyplot(forecast_fig)
        else:
            st.warning("将来予測プロットを生成できませんでした。")

    with tab3:
        st.subheader("将来予測データ")
        forecast_df = results.get("forecast_df")

        if forecast_df is not None and not forecast_df.empty:
            st.dataframe(forecast_df)
            excel_data = to_excel(forecast_df)
            original_filename = results.get("excel_filename", "data")
            download_filename = f"forecast_{original_filename}.xlsx"
            st.download_button(
                label="予測データをExcelとしてダウンロード",
                data=excel_data,
                file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("予測データを表示できませんでした。")


# --- 分析ロジック (旧main.pyから移植) ---


def extract_data_from_uploaded_file(
    uploaded_file,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """アップロードされたExcelファイルからデータを抽出する"""
    try:
        df = pd.read_excel(uploaded_file, header=None)
        if df.shape[1] < 2:
            st.error("Excelファイルには少なくとも2つの列（時間と値）が必要です。")
            return None
        df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
        time_array = df.iloc[:, 0].values
        value_array = df.iloc[:, 1].values
        filename = os.path.splitext(uploaded_file.name)[0]
        return time_array, value_array, filename
    except Exception as e:
        st.error(f"Excelファイルの読み込み中にエラーが発生しました: {e}")
        return None


def run_analysis(settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """ユーザー設定に基づいて分析パイプラインを実行する"""
    uploaded_file = settings.get("uploaded_file")
    if not uploaded_file:
        st.warning("分析するファイルをアップロードしてください。")
        return None

    extraction_result = extract_data_from_uploaded_file(uploaded_file)
    if not extraction_result:
        return None

    time_array, value_array, excel_filename = extraction_result

    try:
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

        fitter = ParameterFitterService(model_params, time_array, value_array)
        best_params, min_sse = fitter.fit_parameters()
        fitting_visualizer = FittingVisualizerService(prediction_settings)
        fitting_fig = fitting_visualizer.plot_with_equation(
            time_array, value_array, fitter.get_fitted_equation()
        )

        predictor = FuturePredictorService(
            fitter.get_fitted_equation(), prediction_settings
        )
        forecast_time_array, forecast_value_array = predictor.predict(
            time_array, value_array
        )
        forecast_visualizer = ForecastVisualizerService(prediction_settings)
        forecast_fig = forecast_visualizer.plot_forecast(
            time_array, value_array, forecast_time_array, forecast_value_array
        )

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
        import traceback

        st.error(f"詳細: {traceback.format_exc()}")
        return None


# --- ページのエントリポイント ---


def analysis_page():
    st.set_page_config(page_title="分析ページ", layout="wide")
    st.title("分析ページ")

    user_settings = render_sidebar()

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    if user_settings["run_analysis"]:
        with st.spinner("分析を実行中..."):
            st.session_state.analysis_results = run_analysis(user_settings)

    if st.session_state.analysis_results:
        render_results(st.session_state.analysis_results)
    else:
        st.info("サイドバーから設定を行い、「分析実行」ボタンを押してください。")


if __name__ == "__main__":
    analysis_page()
