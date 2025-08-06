"""
データ抽出機能
"""
import os
import sys
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


class DataExtractor:
    """
    Excelファイルからデータを抽出するクラス
    """
    
    @staticmethod
    def extract_from_directory(input_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        inputディレクトリからExcelファイル（.xlsx）を1つだけ特定し、データを抽出
        
        Args:
            input_dir (str): 入力ディレクトリのパス
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (時間データの配列, 値データの配列)
        
        Raises:
            FileNotFoundError: Excelファイルが見つからない場合
            ValueError: データ形式が不正な場合
        """
        xlsx_files: List[str] = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
        if not xlsx_files:
            raise FileNotFoundError(f"inputディレクトリにxlsxファイルが見つかりません: {input_dir}")
        
        excel_filename: str = xlsx_files[0]
        excel_path: str = os.path.join(input_dir, excel_filename)
        
        try:
            df: pd.DataFrame = pd.read_excel(excel_path, header=0, engine="openpyxl")
            
            # 列名の確認
            expected_columns = ['time', 'value']
            if list(df.columns[:2]) != expected_columns:
                raise ValueError(f"1行目は 'time', 'value' という列名である必要があります。現在: {list(df.columns[:2])}")
            
            # データ型の確認
            if not (is_numeric_dtype(df['time']) and is_numeric_dtype(df['value'])):
                raise ValueError("'time'列と'value'列は数値データである必要があります")
            
            # データの変換と検証
            time_data = df['time'].to_numpy()
            value_data = df['value'].to_numpy()
            
            # NaNや無限大の値がないかチェック
            if np.any(np.isnan(time_data)) or np.any(np.isnan(value_data)):
                raise ValueError("データにNaN値が含まれています")
            
            if np.any(np.isinf(time_data)) or np.any(np.isinf(value_data)):
                raise ValueError("データに無限大値が含まれています")
            
            # 時間データが昇順かチェック
            if not np.all(np.diff(time_data) >= 0):
                raise ValueError("時間データが昇順でありません")
            
            # 値データが非負かチェック
            if np.any(value_data < 0):
                raise ValueError("値データに負の値が含まれています")
            
            return time_data, value_data, excel_filename
            
        except FileNotFoundError:
            raise FileNotFoundError(f"入力ファイルが見つかりません: {excel_path}")
        except Exception as e:
            raise Exception(f"データ抽出エラー: {e}")
