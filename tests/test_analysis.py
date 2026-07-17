from __future__ import annotations

from flight_delays.analysis import (
    CORRELATION_COLUMNS,
    build_correlations,
    build_overview,
    build_status_distribution,
)


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
    assert overview["delayed_flight_count"] == 3_578
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


def test_excludes_all_post_flight_causes(spark_sample_data) -> None:
    assert {
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    }.isdisjoint(spark_sample_data.columns)
