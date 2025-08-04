"""
ロジスティックモデルの将来予測結果をプロット
"""
import matplotlib.pyplot as plt

# 日本語フォントの設定 (macOS標準のヒラギノ角ゴシック)
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け対策

def plot_forecast_result(t_actual, P_actual, t_forecast, P_forecast, output_path, start_year=1950):
    """
    ロジスティックモデルによる将来予測結果をプロットする
    
    Args:
        t_actual (np.array): 実績データの期間
        P_actual (np.array): 実績データの値
        t_forecast (np.array): 予測時刻の配列
        P_forecast (np.array): 予測値の配列
        output_path (str): 出力画像のファイルパス
        start_year (int, optional): プロットのx軸の開始年. Defaults to 1950.
    """
    # 予測結果の可視化
    plt.figure(figsize=(12, 7))

    # 実績データのプロット (x軸を開始年で調整)
    actual_years = t_actual + start_year
    plt.plot(actual_years, P_actual, 'o', label=f'実際の人口データ ({actual_years[0]}-{actual_years[-1]}年)')

    # 予測結果のプロット
    forecast_years = t_forecast + start_year
    plt.plot(forecast_years, P_forecast, '-', label=f'ロジスティックモデルによる将来予測 ({int(forecast_years[-1])}年まで)')

    plt.title('ロジスティックモデルによる人口推移と将来予測')
    plt.xlabel('年')
    plt.ylabel('人口（千人）')
    # 予測開始時点に垂直線を追加
    plt.axvline(x=actual_years[-1], color='gray', linestyle='--', label=f'予測開始 ({actual_years[-1]}年)')
    plt.legend()
    plt.grid(True)

    # グラフをファイルに保存
    plt.savefig(output_path)
    print(f"予測グラフを '{output_path}' として保存しました。")
    plt.close() # メモリリークを防ぐためにプロットを閉じる
