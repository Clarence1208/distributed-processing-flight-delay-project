from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from flight_delays.parsing import (
    parse_and_validate,
    read_flights_csv,
    sample_rows,
)


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("Tests du projet de retards de vols")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def spark_sample_data(spark: SparkSession):
    """Construit les 2 000 lignes Spark sans dépendre d'une sortie générée."""

    data = parse_and_validate(
        sample_rows(
            read_flights_csv(spark, "data/flight_data_2024_sample.csv"),
            sample_size=2000,
            seed=42,
        )
    ).cache()
    data.count()
    yield data
    data.unpersist()
