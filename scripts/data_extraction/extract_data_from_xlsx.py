import os
import sys
from typing import Tuple, List
import pandas as pd
import numpy as np

def extract_data_from_xlsx(input_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    inputディレクトリからExcelファイル（.xlsx）を1つだけ特定し、1枚目のシートからデータを抽出します。
    データ形式が正しいか確認した上で、配列として返します。

    Args:
        input_dir (str): 入力ディレクトリのパス。

    Returns:
        Tuple[np.ndarray, np.ndarray]: (時間データの配列, 値データの配列)

    使用例:
        extract_data_from_xlsx("input")
    """
    xlsx_files: List[str] = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    if not xlsx_files:
        sys.exit(f"エラー: inputディレクトリにxlsxファイルが見つかりません")
    excel_filename: str = xlsx_files[0]
    excel_path: str = os.path.join(input_dir, excel_filename)
    try:
        df: pd.DataFrame = pd.read_excel(excel_path, header=0, engine="openpyxl")
        # 1行目が 'time', 'value' であることを確認
        expected_columns = ['time', 'value']
        if list(df.columns[:2]) != expected_columns:
            sys.exit(f"エラー: 1行目は 'time', 'value' という列名である必要があります")
        # 両列が数値型であることを確認
        if not (np.issubdtype(df['time'].dtype, np.number) and np.issubdtype(df['value'].dtype, np.number)):
            sys.exit(f"エラー: 'time'列と'value'列は数値データである必要があります")

        t_actual: np.ndarray = df['time'].values
        P_actual: np.ndarray = df['value'].values
        return t_actual, P_actual

    except FileNotFoundError:
        sys.exit(f"エラー: 入力ファイルが見つかりません '{excel_path}'")
    except Exception as e:
        sys.exit(f"エラーが発生しました: {e}")
