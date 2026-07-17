"""Parsing et contrôle qualité des vols avec PySpark.

Le CSV est d'abord lu entièrement comme du texte. Cette stratégie évite que Spark
masque une valeur mal formée en la convertissant silencieusement en valeur nulle.
Les conversions et les règles métier sont ensuite appliquées explicitement.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.storagelevel import StorageLevel


SOURCE_COLUMNS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "fl_date",
    "op_unique_carrier",
    "op_carrier_fl_num",
    "origin",
    "origin_city_name",
    "origin_state_nm",
    "dest",
    "dest_city_name",
    "dest_state_nm",
    "crs_dep_time",
    "dep_time",
    "dep_delay",
    "taxi_out",
    "wheels_off",
    "wheels_on",
    "taxi_in",
    "crs_arr_time",
    "arr_time",
    "arr_delay",
    "cancelled",
    "cancellation_code",
    "diverted",
    "crs_elapsed_time",
    "actual_elapsed_time",
    "air_time",
    "distance",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]

POST_FLIGHT_CAUSE_COLUMNS = {
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
}

PARSED_COLUMNS = [
    name for name in SOURCE_COLUMNS if name not in POST_FLIGHT_CAUSE_COLUMNS
]

INTEGER_COLUMNS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "op_carrier_fl_num",
    "crs_dep_time",
    "dep_time",
    "wheels_off",
    "wheels_on",
    "crs_arr_time",
    "arr_time",
    "cancelled",
    "diverted",
]

FLOAT_COLUMNS = [
    "dep_delay",
    "taxi_out",
    "taxi_in",
    "arr_delay",
    "crs_elapsed_time",
    "actual_elapsed_time",
    "air_time",
    "distance",
]

TEXT_COLUMNS = [
    "op_unique_carrier",
    "origin",
    "origin_city_name",
    "origin_state_nm",
    "dest",
    "dest_city_name",
    "dest_state_nm",
    "cancellation_code",
]

REQUIRED_COLUMNS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "fl_date",
    "op_unique_carrier",
    "op_carrier_fl_num",
    "origin",
    "origin_city_name",
    "origin_state_nm",
    "dest",
    "dest_city_name",
    "dest_state_nm",
    "crs_dep_time",
    "crs_arr_time",
    "cancelled",
    "diverted",
    "crs_elapsed_time",
    "distance",
]

HHMM_COLUMNS = [
    "crs_dep_time",
    "dep_time",
    "wheels_off",
    "wheels_on",
    "crs_arr_time",
    "arr_time",
]

CORRUPT_RECORD_COLUMN = "_corrupt_record"
DEFAULT_SPARK_SAMPLE_SIZE = 10_000


def create_spark_session(
    application_name: str = "Parsing des vols 2024", master: str | None = None
) -> SparkSession:
    """Crée une session Spark ou récupère la session déjà active."""

    builder = (
        SparkSession.builder.appName(application_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
    )
    if master:
        builder = builder.master(master)
    return builder.getOrCreate()


def raw_csv_schema() -> StructType:
    """Retourne le schéma textuel utilisé à l'entrée du CSV."""

    fields = [StructField(name, StringType(), True) for name in SOURCE_COLUMNS]
    fields.append(StructField(CORRUPT_RECORD_COLUMN, StringType(), True))
    return StructType(fields)


def read_flights_csv(spark: SparkSession, path: str) -> DataFrame:
    """Lit un ou plusieurs CSV de vols sans inférence de schéma."""

    return (
        spark.read.format("csv")
        .schema(raw_csv_schema())
        .option("header", "true")
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_RECORD_COLUMN)
        .option("encoding", "UTF-8")
        .option("quote", '"')
        .option("escape", '"')
        .load(path)
    )


def sample_rows(data: DataFrame, sample_size: int, seed: int) -> DataFrame:
    """Sélectionne un nombre exact de lignes aléatoires de façon reproductible."""

    if sample_size <= 0:
        raise ValueError("La taille de l'échantillon doit être strictement positive.")
    return data.orderBy(F.rand(seed)).limit(sample_size)


def _clean_text(name: str) -> Column:
    value = F.trim(F.col(name))
    return F.when(value == "", F.lit(None)).otherwise(value)


def _to_integer(name: str) -> Column:
    """Accepte un entier CSV écrit sous la forme 12 ou 12.0, mais pas 12.5."""

    value = _clean_text(name)
    is_integer = value.rlike(r"^[+-]?\d+(?:\.0+)?$")
    return F.when(is_integer, value.cast("double").cast("int"))


def _to_float(name: str) -> Column:
    value = _clean_text(name)
    return value.cast("double")


def _conversion_for(name: str) -> Column:
    if name in INTEGER_COLUMNS:
        return _to_integer(name)
    if name in FLOAT_COLUMNS:
        return _to_float(name)
    if name == "fl_date":
        return F.to_date(_clean_text(name), "yyyy-MM-dd")
    if name in TEXT_COLUMNS:
        return _clean_text(name)
    raise ValueError(f"Aucune conversion déclarée pour la colonne {name!r}")


def _has_invalid_format(name: str) -> Column:
    has_value = _clean_text(name).isNotNull()
    return has_value & _conversion_for(name).isNull()


def _is_valid_hhmm(column: Column) -> Column:
    """Valide HHMM, en acceptant 2400 comme minuit en fin de journée."""

    return (column == 2400) | (
        column.between(0, 2359) & ((column % F.lit(100)) < 60)
    )


def _hhmm_to_minutes(column: Column) -> Column:
    return F.when(column == 2400, F.lit(0)).otherwise(
        F.floor(column / F.lit(100)) * 60 + (column % F.lit(100))
    ).cast("int")


def _add_validation_errors(
    data: DataFrame, rules: Sequence[tuple[str, Column]]
) -> DataFrame:
    possible_errors = [
        F.when(condition, F.lit(code)) for code, condition in rules
    ]
    return (
        data.withColumn("_possible_errors", F.array(*possible_errors))
        .withColumn(
            "validation_errors",
            F.expr("filter(_possible_errors, error -> error is not null)"),
        )
        .drop("_possible_errors")
        .withColumn(
            "is_valid_row", F.size(F.col("validation_errors")) == F.lit(0)
        )
    )


def parse_and_validate(raw_data: DataFrame) -> DataFrame:
    """Convertit les types et annote chaque ligne avec ses erreurs de qualité."""

    typed_columns = INTEGER_COLUMNS + FLOAT_COLUMNS + ["fl_date"]
    format_check_columns = [f"_invalid_format_{name}" for name in typed_columns]
    data = raw_data.select(
        *[_conversion_for(name).alias(name) for name in PARSED_COLUMNS],
        F.col(CORRUPT_RECORD_COLUMN),
        *[
            _has_invalid_format(name).alias(f"_invalid_format_{name}")
            for name in typed_columns
        ],
    )

    rules: list[tuple[str, Column]] = [
        ("corrupt_csv_row", F.col(CORRUPT_RECORD_COLUMN).isNotNull())
    ]
    rules.extend(
        (f"invalid_format:{name}", F.col(f"_invalid_format_{name}"))
        for name in typed_columns
    )
    rules.extend(
        (f"missing_required_value:{name}", F.col(name).isNull())
        for name in REQUIRED_COLUMNS
    )
    rules.extend(
        [
            ("invalid_range:month", ~F.col("month").between(1, 12)),
            ("invalid_range:day_of_week", ~F.col("day_of_week").between(1, 7)),
            (
                "inconsistent_date",
                (F.col("year") != F.year("fl_date"))
                | (F.col("month") != F.month("fl_date"))
                | (F.col("day_of_month") != F.dayofmonth("fl_date")),
            ),
            ("invalid_flight_number", F.col("op_carrier_fl_num") <= 0),
            ("negative_distance", F.col("distance") < 0),
            ("non_positive_scheduled_duration", F.col("crs_elapsed_time") <= 0),
            ("invalid_flag:cancelled", ~F.col("cancelled").isin(0, 1)),
            ("invalid_flag:diverted", ~F.col("diverted").isin(0, 1)),
            (
                "missing_cancellation_code",
                (F.col("cancelled") == 1) & F.col("cancellation_code").isNull(),
            ),
            (
                "invalid_cancellation_code",
                F.col("cancellation_code").isNotNull()
                & ~F.col("cancellation_code").isin("A", "B", "C", "D"),
            ),
        ]
    )
    rules.extend(
        (
            f"invalid_range:{name}",
            F.col(name).isNotNull() & ~_is_valid_hhmm(F.col(name)),
        )
        for name in HHMM_COLUMNS
    )
    data = _add_validation_errors(data, rules).drop(*format_check_columns)
    return (
        data.withColumn(
            "scheduled_departure_minutes",
            _hhmm_to_minutes(F.col("crs_dep_time")),
        )
        .withColumn(
            "scheduled_arrival_minutes",
            _hhmm_to_minutes(F.col("crs_arr_time")),
        )
    )


def split_rows(validated_data: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Sépare les vols valides des lignes placées en quarantaine."""

    technical_columns = [
        CORRUPT_RECORD_COLUMN,
        "validation_errors",
        "is_valid_row",
    ]
    valid_rows = validated_data.filter("is_valid_row").drop(*technical_columns)
    rejected_rows = validated_data.filter("not is_valid_row")
    return valid_rows, rejected_rows


def write_results(
    validated_data: DataFrame, output_directory: str, mode: str
) -> tuple[int, int]:
    """Écrit le Parquet propre, la quarantaine et le rapport qualité."""

    # Le jeu complet dépasse aisément la mémoire d'un poste local. Le stockage
    # sur disque réserve la mémoire Spark au tri de l'écriture partitionnée.
    validated_data = validated_data.persist(StorageLevel.DISK_ONLY)
    valid_rows, rejected_rows = split_rows(validated_data)
    valid_row_count = valid_rows.count()
    rejected_row_count = rejected_rows.count()

    valid_rows.write.mode(mode).partitionBy("year", "month").parquet(
        f"{output_directory}/flights"
    )
    rejected_rows.write.mode(mode).parquet(f"{output_directory}/rejects")

    spark = validated_data.sparkSession
    global_report = spark.createDataFrame(
        [
            (
                valid_row_count + rejected_row_count,
                valid_row_count,
                rejected_row_count,
            )
        ],
        ["row_count", "valid_row_count", "rejected_row_count"],
    )
    global_report.coalesce(1).write.mode(mode).json(
        f"{output_directory}/quality_report/global"
    )
    (
        rejected_rows.select(F.explode("validation_errors").alias("error"))
        .groupBy("error")
        .count()
        .orderBy(F.desc("count"), "error")
        .coalesce(1)
        .write.mode(mode)
        .json(f"{output_directory}/quality_report/by_error")
    )
    validated_data.unpersist()
    return valid_row_count, rejected_row_count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse et contrôle les données de vols 2024 avec PySpark."
    )
    parser.add_argument(
        "--input",
        default="data/flight_data_2024_sample.csv",
        help="Chemin ou motif Spark vers le ou les CSV sources.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/spark/flights_10000",
        help="Dossier racine des fichiers Parquet et du rapport qualité.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SPARK_SAMPLE_SIZE,
        help="Nombre exact de lignes à sélectionner aléatoirement avant le parsing.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Graine aléatoire utilisée pour obtenir un échantillon reproductible.",
    )
    parser.add_argument(
        "--mode",
        choices=["errorifexists", "overwrite", "append"],
        default="errorifexists",
        help="Comportement si les sorties existent déjà.",
    )
    parser.add_argument(
        "--master",
        default="local[2]",
        help=(
            "Master Spark. La valeur locale limite la concurrence pour éviter "
            "de saturer la mémoire ; utiliser par exemple 'yarn' sur un cluster."
        ),
    )
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    spark = create_spark_session(master=arguments.master)
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw_data = read_flights_csv(spark, arguments.input)
        if arguments.sample_size is not None:
            raw_data = sample_rows(
                raw_data, arguments.sample_size, arguments.sample_seed
            )
        validated_data = parse_and_validate(raw_data)
        valid_row_count, rejected_row_count = write_results(
            validated_data, arguments.output, arguments.mode
        )
        print(
            "Parsing terminé : "
            f"{valid_row_count:,} lignes valides, "
            f"{rejected_row_count:,} lignes rejetées."
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
