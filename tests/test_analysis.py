from __future__ import annotations

from flight_delays.analysis import (
    CORRELATION_COLUMNS,
    build_correlations,
    build_delay_causes,
    build_overview,
    build_status_distribution,
)
from flight_delays.parsing import DELAY_CAUSE_COLUMNS


def test_builds_overview_and_status_distribution(spark_sample_data) -> None:
    overview = build_overview(spark_sample_data).first().asDict()
    status_total = (
        build_status_distribution(spark_sample_data)
        .groupBy()
        .sum("flight_count")
        .first()[0]
    )

    assert overview["flight_count"] == 10_000
    assert overview["completed_flight_count"] <= 10_000
    assert 0 <= overview["delayed_flight_percentage"] <= 100
    assert status_total == 10_000


def test_builds_arrival_delay_correlations(spark_sample_data) -> None:
    matrix, arrival_correlations, correlation_row_count = build_correlations(
        spark_sample_data
    )

    assert matrix.count() == len(CORRELATION_COLUMNS) ** 2
    assert arrival_correlations.count() == len(CORRELATION_COLUMNS) - 1
    assert correlation_row_count > 9_500
    assert (
        arrival_correlations.filter("absolute_correlation > 1").count() == 0
    )
    assert "late_aircraft_delay" not in CORRELATION_COLUMNS


def test_excludes_previous_aircraft_cause(spark_sample_data) -> None:
    causes = build_delay_causes(spark_sample_data)

    assert "late_aircraft_delay" not in DELAY_CAUSE_COLUMNS
    assert causes.filter("cause = 'late_aircraft_delay'").count() == 0
    assert causes.count() == 4
