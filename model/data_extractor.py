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

    def __init__(self, file_path: str):
        """
        DataExtractor の初期化
        
        Args:
            file_path: 抽出するExcelファイルのパス
        """
        self.file_path = file_path
        self.time_data: Optional[np.ndarray] = None
        self.value_data: Optional[np.ndarray] = None
    
    def extract_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        指定されたExcelファイルからデータを抽出
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (時間データの配列, 値データの配列)
        
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: データ形式が不正な場合
        """
        try:
            df: pd.DataFrame = pd.read_excel(self.file_path, header=0, engine="openpyxl")
            
            # 列名の確認
            expected_columns = ['time', 'value']
            if list(df.columns[:2]) != expected_columns:
                raise ValueError(f"1行目は 'time', 'value' という列名である必要があります。現在: {list(df.columns[:2])}")
            
            # データ型の確認
            if not (is_numeric_dtype(df['time']) and is_numeric_dtype(df['value'])):
                raise ValueError("'time'列と'value'列は数値データである必要があります")
            
            # データの保存
            self.time_data = df['time'].to_numpy()
            self.value_data = df['value'].to_numpy()
            
            return self.time_data, self.value_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"入力ファイルが見つかりません: {self.file_path}")
        except Exception as e:
            raise Exception(f"データ抽出エラー: {e}")
    
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
        
        # DataExtractorのインスタンスを作成してデータを抽出
        extractor = DataExtractor(excel_path)
        return extractor.extract_data()
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        抽出済みデータを取得
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (時間データの配列, 値データの配列)
        
        Raises:
            ValueError: データが抽出されていない場合
        """
        if self.time_data is None or self.value_data is None:
            raise ValueError("データが抽出されていません。先にextract_from_file()またはextract_from_directory()を呼び出してください。")
        
        return self.time_data, self.value_data
    
    def validate_data(self) -> bool:
        """
        抽出されたデータの妥当性を検証
        
        Returns:
            bool: データが有効な場合True
        """
        if self.time_data is None or self.value_data is None:
            return False
        
        # NaNや無限大の値がないかチェック
        if np.any(np.isnan(self.time_data)) or np.any(np.isnan(self.value_data)):
            return False
        
        if np.any(np.isinf(self.time_data)) or np.any(np.isinf(self.value_data)):
            return False
        
        # 時間データが昇順かチェック
        if not np.all(np.diff(self.time_data) >= 0):
            return False
        
        # 値データが非負かチェック（人口データの場合）
        if np.any(self.value_data < 0):
            return False
        
        return True
    
    def get_data_info(self) -> dict:
        """
        データの基本情報を取得
        
        Returns:
            dict: データの基本統計情報
        """
        if self.time_data is None or self.value_data is None:
            return {}
        
        return {
            "file_path": self.file_path,
            "data_points": len(self.time_data),
            "time_range": (self.time_data.min(), self.time_data.max()),
            "value_range": (self.value_data.min(), self.value_data.max()),
            "time_mean": self.time_data.mean(),
            "value_mean": self.value_data.mean(),
            "time_std": self.time_data.std(),
            "value_std": self.value_data.std()
        }
