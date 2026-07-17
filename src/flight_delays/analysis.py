"""Analyse descriptive et corrélations du petit dataset Spark."""

from __future__ import annotations

import argparse
import math

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from flight_delays.parsing import create_spark_session

CORRELATION_COLUMNS = [
    "month",
    "day_of_month",
    "day_of_week",
    "scheduled_departure_minutes",
    "scheduled_arrival_minutes",
    "crs_elapsed_time",
    "distance",
    "dep_delay",
    "taxi_out",
    "taxi_in",
    "actual_elapsed_time",
    "air_time",
    "arr_delay",
]

PRE_DEPARTURE_COLUMNS = {
    "month",
    "day_of_month",
    "day_of_week",
    "scheduled_departure_minutes",
    "scheduled_arrival_minutes",
    "crs_elapsed_time",
    "distance",
}


def load_spark_data(spark: SparkSession, path: str) -> DataFrame:
    """Charge le Parquet de l'échantillon réservé à l'analyse Spark."""

    return spark.read.parquet(path)


def _completed_condition() -> Column:
    return (
        (F.col("cancelled") == 0)
        & (F.col("diverted") == 0)
        & F.col("arr_delay").isNotNull()
    )


def _delayed_condition() -> Column:
    return _completed_condition() & (F.col("arr_delay") > 0)


def add_flight_status(data: DataFrame) -> DataFrame:
    """Ajoute un statut exclusif utile aux agrégations."""

    return data.withColumn(
        "flight_status",
        F.when(F.col("cancelled") == 1, F.lit("cancelled"))
        .when(F.col("diverted") == 1, F.lit("diverted"))
        .when(F.col("arr_delay").isNull(), F.lit("unknown"))
        .when(F.col("arr_delay") > 0, F.lit("delayed"))
        .otherwise(F.lit("on_time")),
    )


def build_overview(data: DataFrame) -> DataFrame:
    """Calcule les indicateurs globaux principaux."""

    overview = data.agg(
        F.count(F.lit(1)).alias("flight_count"),
        F.sum(F.when(_completed_condition(), 1).otherwise(0)).alias(
            "completed_flight_count"
        ),
        F.sum(F.when(_delayed_condition(), 1).otherwise(0)).alias(
            "delayed_flight_count"
        ),
        F.sum(F.when(F.col("cancelled") == 1, 1).otherwise(0)).alias(
            "cancelled_flight_count"
        ),
        F.sum(F.when(F.col("diverted") == 1, 1).otherwise(0)).alias(
            "diverted_flight_count"
        ),
        F.avg(F.when(_completed_condition(), F.col("arr_delay"))).alias(
            "average_arrival_delay_minutes"
        ),
        F.percentile_approx(
            F.when(_completed_condition(), F.col("arr_delay")), 0.5
        ).alias("median_arrival_delay_minutes"),
        F.percentile_approx(
            F.when(_completed_condition(), F.col("arr_delay")), 0.9
        ).alias("p90_arrival_delay_minutes"),
        F.max(F.when(_completed_condition(), F.col("arr_delay"))).alias(
            "maximum_arrival_delay_minutes"
        ),
        F.avg(F.col("dep_delay")).alias("average_departure_delay_minutes"),
    )
    return (
        overview.withColumn(
            "delayed_flight_percentage",
            F.col("delayed_flight_count")
            / F.col("completed_flight_count")
            * F.lit(100.0),
        )
        .withColumn(
            "cancelled_flight_percentage",
            F.col("cancelled_flight_count") / F.col("flight_count") * F.lit(100.0),
        )
        .withColumn(
            "diverted_flight_percentage",
            F.col("diverted_flight_count") / F.col("flight_count") * F.lit(100.0),
        )
    )


def build_status_distribution(data: DataFrame) -> DataFrame:
    """Compte les vols par statut et calcule leur proportion."""

    total = data.count()
    return (
        add_flight_status(data)
        .groupBy("flight_status")
        .agg(F.count(F.lit(1)).alias("flight_count"))
        .withColumn(
            "flight_percentage", F.col("flight_count") / F.lit(total) * F.lit(100.0)
        )
        .orderBy(F.desc("flight_count"), "flight_status")
    )


def build_missing_values(data: DataFrame) -> DataFrame:
    """Compte les valeurs manquantes pour chaque colonne."""

    row_count = data.count()
    aggregated = data.agg(
        *[
            F.sum(F.when(F.col(name).isNull(), 1).otherwise(0)).alias(name)
            for name in data.columns
        ]
    ).first()
    rows = [
        (
            name,
            int(aggregated[name]),
            float(aggregated[name]) / row_count * 100.0 if row_count else 0.0,
        )
        for name in data.columns
    ]
    schema = StructType(
        [
            StructField("column", StringType(), False),
            StructField("missing_count", LongType(), False),
            StructField("missing_percentage", DoubleType(), False),
        ]
    )
    return data.sparkSession.createDataFrame(rows, schema).orderBy(
        F.desc("missing_count"), "column"
    )


def build_group_metrics(data: DataFrame, group_column: str) -> DataFrame:
    """Calcule les indicateurs de ponctualité pour une dimension."""

    return (
        data.groupBy(group_column)
        .agg(
            F.count(F.lit(1)).alias("flight_count"),
            F.sum(F.when(_completed_condition(), 1).otherwise(0)).alias(
                "completed_flight_count"
            ),
            F.sum(F.when(_delayed_condition(), 1).otherwise(0)).alias(
                "delayed_flight_count"
            ),
            F.sum(F.when(F.col("cancelled") == 1, 1).otherwise(0)).alias(
                "cancelled_flight_count"
            ),
            F.avg(F.when(_completed_condition(), F.col("arr_delay"))).alias(
                "average_arrival_delay_minutes"
            ),
            F.avg(F.col("dep_delay")).alias("average_departure_delay_minutes"),
        )
        .withColumn(
            "delayed_flight_percentage",
            F.when(
                F.col("completed_flight_count") > 0,
                F.col("delayed_flight_count")
                / F.col("completed_flight_count")
                * F.lit(100.0),
            ),
        )
        .withColumn(
            "cancelled_flight_percentage",
            F.col("cancelled_flight_count") / F.col("flight_count") * F.lit(100.0),
        )
        .orderBy(F.desc("flight_count"), group_column)
    )


def build_correlations(data: DataFrame) -> tuple[DataFrame, DataFrame, int]:
    """Calcule la matrice de Pearson sur les vols achevés sans valeur manquante."""

    correlation_data = (
        data.filter(_completed_condition())
        .select(*CORRELATION_COLUMNS)
        .na.drop(subset=CORRELATION_COLUMNS)
    )
    assembler = VectorAssembler(
        inputCols=CORRELATION_COLUMNS,
        outputCol="features",
        handleInvalid="skip",
    )
    vectors = assembler.transform(correlation_data).select("features").cache()
    correlation_row_count = vectors.count()
    matrix = Correlation.corr(vectors, "features", "pearson").first()[0]

    rows = []
    for row_index, row_name in enumerate(CORRELATION_COLUMNS):
        for column_index, column_name in enumerate(CORRELATION_COLUMNS):
            value = float(matrix[row_index, column_index])
            rows.append(
                (
                    row_name,
                    column_name,
                    value if math.isfinite(value) else None,
                )
            )
    vectors.unpersist()

    schema = StructType(
        [
            StructField("variable_x", StringType(), False),
            StructField("variable_y", StringType(), False),
            StructField("correlation", DoubleType(), True),
        ]
    )
    matrix_data = data.sparkSession.createDataFrame(rows, schema)
    arrival_correlations = (
        matrix_data.filter(
            (F.col("variable_y") == "arr_delay")
            & (F.col("variable_x") != "arr_delay")
        )
        .select(
            F.col("variable_x").alias("variable"),
            "correlation",
        )
        .withColumn("absolute_correlation", F.abs(F.col("correlation")))
        .withColumn(
            "available_before_departure",
            F.col("variable").isin(sorted(PRE_DEPARTURE_COLUMNS)),
        )
        .orderBy(F.desc_nulls_last("absolute_correlation"), "variable")
    )
    return matrix_data, arrival_correlations, correlation_row_count


def build_numeric_summary(data: DataFrame) -> DataFrame:
    """Produit les statistiques descriptives des variables numériques analysées."""

    return data.select(*CORRELATION_COLUMNS).summary(
        "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
    )


def _write_csv(data: DataFrame, path: str, mode: str) -> None:
    data.coalesce(1).write.mode(mode).option("header", "true").csv(path)


def write_analysis(
    data: DataFrame, output_directory: str, mode: str
) -> tuple[DataFrame, DataFrame, int]:
    """Calcule et écrit toutes les tables de l'analyse Spark."""

    overview = build_overview(data)
    status_distribution = build_status_distribution(data)
    missing_values = build_missing_values(data)
    monthly_metrics = build_group_metrics(data, "month").orderBy("month")
    carrier_metrics = build_group_metrics(data, "op_unique_carrier")
    origin_metrics = build_group_metrics(data, "origin")
    numeric_summary = build_numeric_summary(data)
    correlation_matrix, arrival_correlations, correlation_row_count = (
        build_correlations(data)
    )

    overview.coalesce(1).write.mode(mode).json(f"{output_directory}/overview")
    _write_csv(status_distribution, f"{output_directory}/flight_status", mode)
    _write_csv(missing_values, f"{output_directory}/missing_values", mode)
    _write_csv(monthly_metrics, f"{output_directory}/monthly_metrics", mode)
    _write_csv(carrier_metrics, f"{output_directory}/carrier_metrics", mode)
    _write_csv(origin_metrics, f"{output_directory}/origin_metrics", mode)
    _write_csv(numeric_summary, f"{output_directory}/numeric_summary", mode)
    _write_csv(
        correlation_matrix,
        f"{output_directory}/correlations/matrix",
        mode,
    )
    _write_csv(
        arrival_correlations,
        f"{output_directory}/correlations/with_arrival_delay",
        mode,
    )
    return overview, arrival_correlations, correlation_row_count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse les 10 000 vols réservés à Spark."
    )
    parser.add_argument(
        "--input",
        default="data/processed/spark/flights_10000/flights",
        help="Dossier Parquet produit par l'étape 1.",
    )
    parser.add_argument(
        "--output",
        default="data/analysis/spark",
        help="Dossier dans lequel écrire les résultats de l'analyse.",
    )
    parser.add_argument(
        "--mode",
        choices=["errorifexists", "overwrite"],
        default="errorifexists",
        help="Comportement si les résultats existent déjà.",
    )
    parser.add_argument(
        "--master",
        default="local[2]",
        help="Master Spark utilisé pour l'analyse.",
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    spark = create_spark_session(
        application_name="Analyse des retards de vols", master=arguments.master
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        data = load_spark_data(spark, arguments.input).cache()
        overview, arrival_correlations, correlation_row_count = write_analysis(
            data, arguments.output, arguments.mode
        )
        summary = overview.first().asDict()
        print(f"Analyse terminée sur {summary['flight_count']:,} vols.")
        print(
            "Vols arrivés en retard : "
            f"{summary['delayed_flight_count']:,} "
            f"({summary['delayed_flight_percentage']:.1f} % des vols achevés)."
        )
        print(
            "Lignes utilisées pour les corrélations : "
            f"{correlation_row_count:,}."
        )
        print("Principales corrélations avec arr_delay :")
        for row in arrival_correlations.limit(5).collect():
            print(f"- {row.variable}: {row.correlation:.3f}")
        data.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
