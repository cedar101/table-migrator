#!/usr/bin/env python
"""
table_migrate.py
================

Reads data from a source database DSN and writes to a destination database DSN
"""

from typing import Any
from collections.abc import Iterable
from functools import partial
import warnings

from sqlalchemy import insert
from sqlalchemy.engine import create_engine, Engine

# from pandas.io.sql import SQLTable
from pydantic import BaseModel, AnyUrl, AliasChoices, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    CliApp,
    CliSettingsSource,
)
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich_argparse import RichHelpFormatter
from loguru import logger

# NumPy 임포트시 발생하는 UserWarning 무시
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
    import pandas as pd


type Json = dict[str, Any]


class TableInfo(BaseModel):
    table: str = Field(description="Name of SQL table in database")
    schema_: str = Field(
        validation_alias=AliasChoices("schema"),
        description="Name of SQL schema in database to query",
    )
    columns: list[str] | None = Field(
        validation_alias=AliasChoices("columns", "cols"),
        default=None,
        description="List of column names to select from SQL table",
    )
    index_col: list[str] | None = Field(
        validation_alias=AliasChoices(
            "index_col", "pk", "id", "pk_column", "pk_col", "id_column", "id_col"
        ),
        default="id",
        description="Column to set as unique index (primary key)",
    )
    dsn: AnyUrl = Field(
        validation_alias=AliasChoices("dsn", "uri", "url"),
        description="Data source name (DSN) URL/URI for SQLAlchemy",
    )
    connect_args: Json | None = Field(
        validation_alias=AliasChoices("connect_args", "args"),
        default=None,
        description="A dictionary of options which will be passed directly to the DBAPI’s connect() method as additional keyword arguments",
    )


class Settings(BaseSettings):
    chunk_size: int = Field(
        validation_alias=AliasChoices("chunk_size", "chunk", "c"),
        default=10000,
        description="Specify the number of rows in each batch to be written at a time.",
    )
    source: TableInfo = Field(
        validation_alias=AliasChoices("source", "src", "s", "from", "f")
    )
    destination: TableInfo = Field(
        validation_alias=AliasChoices("dest", "destination", "d", "to", "t")
    )
    df_query: str | None = Field(
        default=None,
        validation_alias=AliasChoices("df_query", "query", "filter", "q"),
        description="DataFrame.query() method expression.",
    )

    model_config = SettingsConfigDict(
        loc_by_alias=False,
        hide_input_in_errors=True,
        cli_parse_args=True,
        cli_avoid_json=True,
        cli_exit_on_error=False,
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
    )

    def cli_cmd(self):
        logger.info(f"Settings: {self.model_dump()}")

        source_conn = create_engine(str(self.source.dsn))
        destination_conn = create_engine(
            str(self.destination.dsn),
            connect_args=self.destination.connect_args,
            echo_pool=True,
        )

        logger.info("loading source table...")
        df = pd.read_sql_table(
            schema=self.source.schema_,
            table_name=self.source.table,
            columns=self.source.columns if self.source.columns else None,
            index_col=self.source.index_col if self.source.index_col else None,
            con=source_conn,
        )
        if self.df_query is not None:
            df = df.query(self.df_query)

        total_records = len(df)
        logger.info(f"{total_records = }")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task(
                "[green]Inserting destination table...", total=total_records
            )
            df.to_sql(
                schema=self.destination.schema_,
                name=self.destination.table,
                con=destination_conn,
                chunksize=self.chunk_size,
                method=partial(execute_insert_multi, progress=progress),
                if_exists="append",
                index=(
                    self.destination.index_col if self.destination.index_col else False
                ),
                index_label=(
                    self.destination.index_col if self.destination.index_col else None
                ),
            )


Settings.__doc__ = __doc__.splitlines()[-1]


def execute_insert_multi(
    table,
    conn: Engine,
    keys: list[str],
    data_iter: Iterable,
    progress=None,
) -> int:
    """
    Multi-value INSERT for DBs support.

    Note: multi-value insert is usually faster for analytics DBs
    and tables containing a few columns
    but performance degrades quickly with increase of columns.

    Source: https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/io/sql.py
    """
    data = [dict(zip(keys, row)) for row in data_iter]
    data_len = len(data)
    stmt = insert(table.table).values(data)
    result = conn.execute(stmt)
    row_count = result.rowcount
    if progress is not None:
        progress.advance(progress.task_ids[0], data_len if row_count < 0 else row_count)
        # progress_bar.update(data_len if row_count < 0 else row_count)
    return row_count


def main():
    try:
        CliApp.run(
            Settings,
            cli_settings_source=CliSettingsSource(
                Settings, formatter_class=RichHelpFormatter
            ),
        )
    except ValidationError as e:
        print(e)


if __name__ == "__main__":
    main()
