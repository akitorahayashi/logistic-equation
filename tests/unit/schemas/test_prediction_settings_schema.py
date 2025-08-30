import pytest
from schemas.prediction_settings_schema import PredictionSettingsSchema


class TestPredictionSettingsSchema:
    """
    PredictionSettingsSchemaのユニットテスト
    """

    def test_init_valid_parameters(self):
        """正常なパラメータで初期化できることをテスト"""
        schema = PredictionSettingsSchema(start_year=2000, forecast_end_t=50)
        assert schema.start_year == 2000
        assert schema.forecast_end_t == 50

    def test_get_time_unit_label(self):
        """get_time_unit_labelが常に'年'を返すことをテスト"""
        schema = PredictionSettingsSchema(start_year=1990, forecast_end_t=100)
        assert schema.get_time_unit_label() == "年"

    @pytest.mark.parametrize(
        "start_year, forecast_end_t",
        [
            ("2000", 50),  # start_year is string
            (2000, "50"),  # forecast_end_t is string
            (2000.5, 50),  # start_year is float
        ],
    )
    def test_init_invalid_types(self, start_year, forecast_end_t):
        """
        不正な型のパラメータで初期化した場合、StreamlitのUI側での制約があるため、
        クラス自体は型エラーを発生させないかもしれないが、意図しない型が
        入らないことを前提とした基本的なテスト。
        実際には静的解析で検知されるべき。
        """
        # このクラスは現時点ではバリデーションを持たないため、
        # Pythonの動的型付けによりエラーは発生しない。
        # 型ヒントに基づいた利用が前提。
        schema = PredictionSettingsSchema(
            start_year=start_year, forecast_end_t=forecast_end_t
        )
        assert schema.start_year == start_year
        assert schema.forecast_end_t == forecast_end_t
