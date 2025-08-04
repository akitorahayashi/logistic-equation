import os
from typing import Final
import numpy as np

# ファイルパス関連
INPUT_DIR: Final[str] = 'input'
CACHE_DIR: Final[str] = '.cache'
OUTPUT_DIR: Final[str] = 'output'
OUTPUT_MD: Final[str] = os.path.join(CACHE_DIR, 'india_population_table.md')
FIT_RESULT_PNG: Final[str] = os.path.join(OUTPUT_DIR, 'fit_result.png')
FORECAST_RESULT_PNG: Final[str] = os.path.join(OUTPUT_DIR, 'forecast_result.png')

# データ読み込み設定
HEADER_ROW: Final[int] = 1      # ヘッダー行のインデックス
TIME_COL: Final[str] = 'time'   # 時間データの列名
VALUE_COL: Final[str] = 'value' # 人口データの列名

# モデルパラメータ
K_RANGE: Final[np.ndarray] = np.arange(2000000, 3000000, 10000)
GAMMA_RANGE: Final[np.ndarray] = np.arange(0.02, 0.04, 0.0005)

# 予測設定
FORECAST_END_T: Final[int] = 250  # 予測の最終期間 (t=250は西暦2200年に相当)
START_YEAR: Final[int] = 1950     # 実績データの開始年
