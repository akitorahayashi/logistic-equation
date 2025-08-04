import os
import sys
from typing import Tuple, List
import pandas as pd
import numpy as np

def excel_to_markdown(input_dir: str, md_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    inputディレクトリからExcelファイル（.xlsx）を1つだけ特定し、Markdown形式のテーブルに変換して保存します。

    Args:
        input_dir (str): 入力ディレクトリのパス。
        md_path (str): 出力するMarkdownファイルのパス（ファイル名・保存先）。

    Returns:
        Tuple[np.ndarray, np.ndarray, str]: (時間データ, 値データ, Excelファイルパス)

    使用例:
        excel_to_markdown(
            "input",  # 入力ディレクトリ
            "output/india_population_table.md"  # 出力Markdownファイル
        )
    """
    xlsx_files: List[str] = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    if not xlsx_files:
        sys.exit(f"エラー: inputディレクトリにxlsxファイルが見つかりません")
    excel_path: str = os.path.join(input_dir, xlsx_files[0])
    try:
        df: pd.DataFrame = pd.read_excel(excel_path, header=None, engine="openpyxl")
        # Markdownテーブルのヘッダーを作成 (A, B, C, ...)
        header: List[str] = [''] + [chr(ord('A') + i) for i in range(df.shape[1])]
        md_lines: List[str] = ['| ' + ' | '.join(header) + ' |',
                               '| ' + ' | '.join(['---'] * len(header)) + ' |']

        # 各行をMarkdown形式に変換
        for idx, row in df.iterrows():
            # 行番号を追加し、全セルを文字列に変換
            line: List[str] = [str(idx+1)] + [str(cell) for cell in row]
            md_lines.append('| ' + ' | '.join(line) + ' |')

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        print(f"'{excel_path}' を '{md_path}' に正常に変換しました。")

        # 1列目: 時間, 2列目: 値（仮定）
        t_actual: np.ndarray = df.iloc[:, 0].values
        P_actual: np.ndarray = df.iloc[:, 1].values
        return t_actual, P_actual, excel_path

    except FileNotFoundError:
        sys.exit(f"エラー: 入力ファイルが見つかりません '{excel_path}'")
    except Exception as e:
        sys.exit(f"エラーが発生しました: {e}")
