import os
from typing import Final

# ファイルパス設定
INPUT_DIR: Final[str] = 'input'
CACHE_DIR: Final[str] = '.cache'
OUTPUT_DIR: Final[str] = 'output'
FIT_RESULT_PNG: Final[str] = os.path.join(OUTPUT_DIR, 'fit_result.png')
FORECAST_RESULT_PNG: Final[str] = os.path.join(OUTPUT_DIR, 'forecast_result.png')

# データ読み込み設定
HEADER_ROW: Final[int] = 1      # ヘッダー行のインデックス
TIME_COL: Final[str] = 'time'   # 時間データの列名
VALUE_COL: Final[str] = 'value' # 時間に対する値データの列名
