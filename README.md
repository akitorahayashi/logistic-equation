## 概要

このプロジェクトは、時系列データをロジスティック方程式に当てはめて分析し、将来予測を行うためのPythonツールです。人口の推移や製品の普及率など、S字型の成長を示す現象のモデリングに利用できます。

Excelファイルからデータを読み込み、最適なパラメータを自動で探索してモデルを構築し、結果をグラフとして可視化します。

## 主な機能

- **Excelからのデータ抽出**: `input`ディレクトリ内のExcelファイルから時系列データを自動で読み込みます。
- **パラメータ自動探索**: 残差平方和（SSE）が最小となるロジスティック方程式のパラメータ（環境収容力 `K`、成長率 `γ`）を探索します。
- **高精度な数値解析**: 4次ルンゲ・クッタ法を用いてロジスティック微分方程式を解き、精度の高いモデルを構築します。
- **結果の可視化**: 元データとフィッティングした曲線、および将来予測を`matplotlib`で描画し、画像ファイルとして出力します。
- **進捗表示**: コンソールに分析の進捗状況を分かりやすく表示します。

## 動作要件

- Python: `==3.12.4`
- Library:
  - `numpy`
  - `pandas`
  - `openpyxl`
  - `matplotlib`
  - `scikit-learn`
  - `yaspin`

## ディレクトリ構成

```
.
├── config/
├── input/
│   └── サンプルデータ.xlsx
├── model/
│   ├── data_extractor.py
│   ├── logistic_equation.py
│   ├── parameter_fitting.py
│   ├── predictor.py
│   └── visualizer.py
├── output/
│   ├── fit_result.png
│   └── forecast_result.png
├── tests/
├── main.py
├── pyproject.toml
└── README.md
```

## インストール

本プロジェクトは[Poetry](https://python-poetry.org/)によるパッケージ管理を前提としています。

```bash
poetry install
```

## 使い方

### 1. 入力データの準備

1.  `input`ディレクトリに、分析したいデータを含むExcelファイル（`.xlsx`形式）を1つ配置します。
2.  Excelファイルの1行目には、必ず`time`と`value`というヘッダーを設定してください。
    - `time`: 時間（年など）を表す数値
    - `value`: `time`に対応する観測値

サンプルとして`input/サンプルデータ.xlsx`が同梱されています。

### 2. スクリプトの実行

プロジェクトのルートディレクトリで以下のコマンドを実行すると、分析パイプラインが開始されます。

```bash
poetry run python main.py
```

## 設定

モデルのパラメータや予測期間は`main.py`スクリプトの冒頭で変更できます。

```python
# main.py

def main() -> None:
    """
    ロジスティック方程式分析パイプラインを実行するメインスクリプト
    """
    # 設定の初期化
    model_params = ModelParameters(
        k_min=2000000.0,
        k_max=3000000.0,
        k_step=10000.0,
        gamma_min=0.02,
        gamma_max=0.04,
        gamma_step=0.0005
    )
    prediction_settings = PredictionSettings(
        start_year=1950,
        forecast_end_t=250
    )
    # ...
```

- `ModelParameters`: パラメータ（`K`, `gamma`）の探索範囲とステップ幅を定義します。
- `PredictionSettings`: 分析の開始年や、将来予測を行う期間の長さを定義します。

## 出力

分析が完了すると、`output`ディレクトリに以下の2つのPNGファイルが生成されます。

- `fit_result.png`: 元のデータ（散布図）と、それに最もフィットするよう計算されたロジスティック曲線（実線）をプロットしたグラフです。
- `forecast_result.png`: 元のデータと、構築したモデルに基づいて将来の値を予測した結果をプロットしたグラフです。
