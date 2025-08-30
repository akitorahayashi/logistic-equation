import streamlit as st
import pandas as pd
import matplotlib.figure
from typing import Dict, Any
from io import BytesIO

def to_excel(df: pd.DataFrame) -> bytes:
    """
    DataFrameをインメモリのExcelファイルに変換する。
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Prediction')
    processed_data = output.getvalue()
    return processed_data

def render_results(results: Dict[str, Any]) -> None:
    """
    分析結果をStreamlitのUIにレンダリングする。

    Args:
        results (Dict[str, Any]): 分析から得られた結果を含む辞書。
                                 - 'best_params': 最適パラメータ (dict)
                                 - 'min_sse': 最小二乗誤差 (float)
                                 - 'fitting_fig': 適合プロットのFigureオブジェクト
                                 - 'forecast_fig': 将来予測プロットのFigureオブジェクト
                                 - 'forecast_df': 予測結果のDataFrame
                                 - 'excel_filename': 元のExcelファイル名
    """
    st.header("📈 分析結果")

    tab1, tab2, tab3 = st.tabs(["フィッティング結果", "将来予測", "予測データ"])

    with tab1:
        st.subheader("最適なパラメータ")

        # 最適パラメータをメトリックとして表示
        best_params = results.get('best_params', {})
        min_sse = results.get('min_sse', 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("成長率 (γ)", f"{best_params.get('gamma', 0):.4f}")

        # Kの値を億や兆に変換して表示
        k_value = best_params.get('K', 0)
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
        fitting_fig = results.get('fitting_fig')
        if fitting_fig:
            st.pyplot(fitting_fig)
        else:
            st.warning("適合プロットを生成できませんでした。")

    with tab2:
        st.subheader("将来予測のプロット")
        forecast_fig = results.get('forecast_fig')
        if forecast_fig:
            st.pyplot(forecast_fig)
        else:
            st.warning("将来予測プロットを生成できませんでした。")

    with tab3:
        st.subheader("将来予測データ")
        forecast_df = results.get('forecast_df')

        if forecast_df is not None and not forecast_df.empty:
            st.dataframe(forecast_df)

            # ダウンロードボタン
            excel_data = to_excel(forecast_df)
            original_filename = results.get('excel_filename', 'data')
            download_filename = f"forecast_{original_filename}"

            st.download_button(
                label="予測データをExcelとしてダウンロード",
                data=excel_data,
                file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("予測データを表示できませんでした。")
