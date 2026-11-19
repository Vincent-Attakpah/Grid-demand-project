import datetime as dt

import pandas as pd

from nowcaster.ingest.clean import read_balance_file
from nowcaster.ingest.eia_bulk import file_names, is_current_half

HEADER = ('"Balancing Authority","Data Date","Hour Number","Local Time at End of Hour",'
          '"UTC Time at End of Hour","Demand Forecast (MW)","Demand (MW)","Net Generation (MW)",'
          '"Total Interchange (MW)","Sum(Valid DIBAs) (MW)","Demand (MW) (Imputed)",'
          '"Net Generation (MW) (Imputed)","Total Interchange (MW) (Imputed)","Demand (MW) (Adjusted)",'
          '"Net Generation (MW) (Adjusted)","Total Interchange (MW) (Adjusted)","Region"')
ROWS = [
    'TVA,01/01/2021,1,01/01/2021 1:00:00 AM,01/01/2021 7:00:00 AM,14289,15105,15400,295,284,,,,15105,15400,295,TEN',
    'TVA,01/01/2021,13,01/01/2021 1:00:00 PM,01/01/2021 7:00:00 PM,"14,000",,15400,295,284,14000,,,14000,15400,295,TEN',
    'AEC,01/01/2021,1,01/01/2021 1:00:00 AM,01/01/2021 7:00:00 AM,679,373,475,102,102,,,,373,475,102,SE',
]


def test_read_balance_file(tmp_path):
    p = tmp_path / "EIA930_BALANCE_2021_Jan_Jun.csv"
    p.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    df = read_balance_file(p)
    assert list(df["ba"]) == ["TVA", "TVA"]  # non-target BA dropped
    assert df["period_utc"].iloc[0] == pd.Timestamp("2021-01-01 07:00", tz="UTC")
    assert df["period_utc"].iloc[1] == pd.Timestamp("2021-01-01 19:00", tz="UTC")  # PM parsed
    assert df["da_forecast"].iloc[1] == 14000  # thousands separator handled
    assert pd.isna(df["demand"].iloc[1]) and df["demand_imputed"].iloc[1]
    assert not df["demand_imputed"].iloc[0]
    assert df["demand_adjusted"].iloc[1] == 14000
    assert df["source_file"].iloc[0] == p.name


def test_file_names_stop_at_today():
    names = file_names(2025, 2026, today=dt.date(2026, 3, 1))
    assert names == ["EIA930_BALANCE_2025_Jan_Jun.csv", "EIA930_BALANCE_2025_Jul_Dec.csv",
                     "EIA930_BALANCE_2026_Jan_Jun.csv"]


def test_is_current_half():
    assert is_current_half("EIA930_BALANCE_2026_Jul_Dec.csv", dt.date(2026, 11, 18))
    assert not is_current_half("EIA930_BALANCE_2026_Jan_Jun.csv", dt.date(2026, 11, 18))
