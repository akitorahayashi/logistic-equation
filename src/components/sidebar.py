import streamlit as st
from typing import Dict, Any, Optional
from io import BytesIO


def render_sidebar() -> Dict[str, Any]:
    """
    Streamlitアプリケーションのサイドバーをレンダリングし、ユーザーの入力を収集する。

    Returns:
        Dict[str, Any]: ユーザーが設定したパラメータとアップロードしたファイルを含む辞書。
                         'run_analysis'キーは分析実行ボタンが押されたかどうかを示す。
    """
    st.sidebar.title("ロジスティック方程式分析ツール")
    st.sidebar.markdown(
        """
    このツールは、アップロードされた時系列データに対してロジスティック方程式の最適なパラメータを探索し、将来予測を行います。
    """
    )

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
        # 単位セレクター
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

    # 実際の値に変換
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
    run_analysis = st.sidebar.button("分析実行", disabled=(uploaded_file is None))
    settings["run_analysis"] = run_analysis

    if uploaded_file is None:
        st.sidebar.warning("分析を開始するには、ファイルをアップロードしてください。")

    return settings
