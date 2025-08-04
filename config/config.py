import os
import numpy as np

# ファイルパス関連
INPUT_DIR = 'input'
CACHE_DIR = '.cache'
OUTPUT_DIR = 'output'
OUTPUT_MD = os.path.join(CACHE_DIR, 'india_population_table.md')
FIT_RESULT_PNG = os.path.join(OUTPUT_DIR, 'fit_result.png')
FORECAST_RESULT_PNG = os.path.join(OUTPUT_DIR, 'forecast_result.png')

# データ読み込み設定
HEADER_ROW = 1      # ヘッダー行のインデックス
TIME_COL = 'time'   # 時間データの列名
VALUE_COL = 'value' # 人口データの列名

# モデルパラメータ
K_RANGE = np.arange(2000000, 3000000, 10000)
GAMMA_RANGE = np.arange(0.02, 0.04, 0.0005)

# 予測設定
FORECAST_END_T = 250  # 予測の最終期間 (t=250は西暦2200年に相当)
START_YEAR = 1950     # 実績データの開始年
