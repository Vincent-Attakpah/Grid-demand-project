from zoneinfo import ZoneInfo

from nowcaster.config import load_bas, weather_settings


def test_bas_load():
    bas = load_bas(include_logging_only=True)
    assert {"SOCO", "TVA", "DUK", "FPL", "ERCO"} <= set(bas)
    for ba in bas.values():
        ZoneInfo(ba.tz)  # valid zone


def test_city_corrections():
    bas = load_bas()
    assert "Huntsville" not in {c.name for c in bas["SOCO"].cities}
    assert "Huntsville" in {c.name for c in bas["TVA"].cities}
    assert "Raleigh" not in {c.name for c in bas["DUK"].cities}


def test_fixed_weather_model():
    assert weather_settings()["model"] != "best_match"
