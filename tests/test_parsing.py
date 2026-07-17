from __future__ import annotations

import csv

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, IntegerType

from flight_delays.parsing import (
    SOURCE_COLUMNS,
    parse_and_validate,
    read_flights_csv,
    sample_rows,
    split_rows,
)

def test_parses_spark_sample(spark_sample_data) -> None:
    assert spark_sample_data.count() == 2_000
    assert spark_sample_data.filter("not is_valid_row").count() == 0
    assert isinstance(spark_sample_data.schema["fl_date"].dataType, DateType)
    assert isinstance(
        spark_sample_data.schema["op_carrier_fl_num"].dataType, IntegerType
    )
    assert (
        spark_sample_data.filter("scheduled_departure_minutes is null").count()
        == 0
    )
    assert "late_aircraft_delay" not in spark_sample_data.columns


def test_random_sample_is_exact_and_reproducible(spark: SparkSession) -> None:
    data = spark.range(100)

    first_sample = [
        row.id for row in sample_rows(data, sample_size=20, seed=42).collect()
    ]
    second_sample = [
        row.id for row in sample_rows(data, sample_size=20, seed=42).collect()
    ]

    assert len(first_sample) == 20
    assert first_sample == second_sample
    assert len(set(first_sample)) == 20


def test_quarantines_invalid_time(spark: SparkSession, tmp_path) -> None:
    path = tmp_path / "invalid_flight.csv"
    row = [
        "2024",
        "1",
        "1",
        "1",
        "2024-01-01",
        "AA",
        "148.0",
        "CLT",
        "Charlotte, NC",
        "North Carolina",
        "PHX",
        "Phoenix, AZ",
        "Arizona",
        "1260",
        "1250.0",
        "-10.0",
        "14.0",
        "1304.0",
        "1500.0",
        "6.0",
        "1523",
        "1506.0",
        "-17.0",
        "0",
        "",
        "0",
        "286.0",
        "273.0",
        "253.0",
        "1773.0",
        "0",
        "0",
        "0",
        "0",
        "0",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SOURCE_COLUMNS)
        writer.writerow(row)

    data = parse_and_validate(read_flights_csv(spark, str(path)))
    valid_rows, rejected_rows = split_rows(data)

    assert valid_rows.count() == 0
    assert rejected_rows.count() == 1
    errors = rejected_rows.select("validation_errors").first()[0]
    assert "invalid_range:crs_dep_time" in errors


def test_distinguishes_invalid_format_from_missing_value(
    spark: SparkSession, tmp_path
) -> None:
    path = tmp_path / "invalid_format.csv"
    row = [
        "2024",
        "january",
        "1",
        "1",
        "2024-01-01",
        "AA",
        "148.0",
        "CLT",
        "Charlotte, NC",
        "North Carolina",
        "PHX",
        "Phoenix, AZ",
        "Arizona",
        "1252",
        "",
        "",
        "",
        "",
        "",
        "",
        "1523",
        "",
        "",
        "1",
        "B",
        "0",
        "286.0",
        "",
        "",
        "1773.0",
        "0",
        "0",
        "0",
        "0",
        "0",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SOURCE_COLUMNS)
        writer.writerow(row)

    data = parse_and_validate(read_flights_csv(spark, str(path)))
    errors = data.select("validation_errors").first()[0]

    assert "invalid_format:month" in errors
    assert "missing_required_value:month" in errors
    assert "invalid_format:dep_time" not in errors
