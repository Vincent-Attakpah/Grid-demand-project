import datetime as dt

import pandas as pd

from nowcaster.timeutils import (horizon_hours, operating_day, operating_day_hours,
                                 remaining_day_hours)


def test_he24_belongs_to_previous_day():
    # Local midnight Jan 2 in New York = 05:00 UTC; that's HE24 of Jan 1.
    assert operating_day(pd.Timestamp("2025-01-02 05:00", tz="UTC"), "America/New_York") == dt.date(2025, 1, 1)
    assert operating_day(pd.Timestamp("2025-01-02 06:00", tz="UTC"), "America/New_York") == dt.date(2025, 1, 2)


def test_dst_day_lengths():
    assert len(operating_day_hours(dt.date(2025, 3, 9), "America/Chicago")) == 23
    assert len(operating_day_hours(dt.date(2025, 11, 2), "America/Chicago")) == 25
    assert len(operating_day_hours(dt.date(2025, 6, 1), "America/Chicago")) == 24


def test_remaining_day_hours():
    tau = pd.Timestamp("2025-01-15 15:20", tz="UTC")  # 09:20 Central
    hrs = remaining_day_hours(tau, "America/Chicago")
    assert hrs[0] == pd.Timestamp("2025-01-15 16:00", tz="UTC")  # HE10 local
    assert hrs[-1] == pd.Timestamp("2025-01-16 06:00", tz="UTC")  # HE24 local
    assert (hrs > tau).all()


def test_horizon():
    assert horizon_hours(pd.Timestamp("2025-01-15 15:20", tz="UTC"), pd.Timestamp("2025-01-15 16:00", tz="UTC")) == 1
    assert horizon_hours(pd.Timestamp("2025-01-15 15:00", tz="UTC"), pd.Timestamp("2025-01-15 21:00", tz="UTC")) == 6
