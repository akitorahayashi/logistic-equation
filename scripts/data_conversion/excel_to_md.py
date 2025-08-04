import pandas as pd
import sys

def excel_to_markdown(excel_path, md_path):
    """
    ExcelファイルをMarkdown形式のテーブルに変換して保存します。

    Args:
        excel_path (str): 入力するExcelファイルのパス。
        md_path (str): 出力するMarkdownファイルのパス（ファイル名・保存先）。

    使用例:
        excel_to_markdown(
            "input/sample.xlsx",  # 入力Excelファイル
            "output/india_population_table.md"  # 出力Markdownファイル
        )

    ※ 入出力ファイル名・保存場所は引数で自由に指定してください。
    """
    try:
        df = pd.read_excel(excel_path, header=None)
        # Markdownテーブルのヘッダーを作成 (A, B, C, ...)
        header = [''] + [chr(ord('A') + i) for i in range(df.shape[1])]
        md_lines = ['| ' + ' | '.join(header) + ' |',
                    '| ' + ' | '.join(['---'] * len(header)) + ' |']

        # 各行をMarkdown形式に変換
        for idx, row in df.iterrows():
            # 行番号を追加し、全セルを文字列に変換
            line = [str(idx+1)] + [str(cell) for cell in row]
            md_lines.append('| ' + ' | '.join(line) + ' |')

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        print(f"'{excel_path}' を '{md_path}' に正常に変換しました。")

    except FileNotFoundError:
        sys.exit(f"エラー: 入力ファイルが見つかりません '{excel_path}'")
    except Exception as e:
        sys.exit(f"エラーが発生しました: {e}")
