"""Helper utilities for HANA integration tests."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from hdbcli import dbapi


class HanaTestUtils:
    @staticmethod
    def execute_sql(
        conn: dbapi.Connection,
        statement: str,
        parameters: Sequence[tuple[Any, ...]] | None = None,
        return_result: bool = False,
    ) -> str | int | None:
        cursor = conn.cursor()
        res = None
        if parameters is not None:
            cursor.executemany(statement, parameters)
        else:
            cursor.execute(statement)
        if return_result:
            res = cursor.fetchone()
        cursor.close()
        conn.commit()
        if return_result and res:
            return res[0]
        return None

    @staticmethod
    def drop_schema_if_exists(conn: dbapi.Connection, schema_name: str) -> None:
        res = HanaTestUtils.execute_sql(
            conn,
            f"SELECT COUNT(*) FROM SYS.SCHEMAS WHERE SCHEMA_NAME='{schema_name}'",
            return_result=True,
        )
        if res != 0:
            HanaTestUtils.execute_sql(conn, f'DROP SCHEMA "{schema_name}" CASCADE')

    @staticmethod
    def drop_old_test_schemas(conn: dbapi.Connection, schema_prefix: str) -> None:
        cursor = conn.cursor()
        try:
            sql = f"""SELECT SCHEMA_NAME FROM SYS.SCHEMAS WHERE SCHEMA_NAME
                      LIKE '{schema_prefix.replace('_', '__')}__%' ESCAPE '_' AND
                      LOCALTOUTC(CREATE_TIME) < ?"""
            cursor.execute(sql, (datetime.now() - timedelta(days=1),))
            rows = cursor.fetchall()

            for row in rows:
                HanaTestUtils.execute_sql(conn, f'DROP SCHEMA "{row[0]}" CASCADE')
        except Exception as ex:
            raise RuntimeError(f"Unable to drop old test schemas. Error: {ex}")
        finally:
            cursor.close()

    @staticmethod
    def generate_schema_name(conn: dbapi.Connection, prefix: str) -> str:
        sql = (
            "SELECT REPLACE(CURRENT_UTCDATE, '-', '') || '_' || BINTOHEX(SYSUUID) "
            "FROM DUMMY;"
        )
        uid = HanaTestUtils.execute_sql(conn, sql, return_result=True)
        return f"{prefix}_{uid}"

    @staticmethod
    def create_and_set_schema(conn: dbapi.Connection, schema_name: str) -> None:
        # HanaTestUtils.dropSchemaIfExists(conn, schema_name)
        HanaTestUtils.execute_sql(conn, f'CREATE SCHEMA "{schema_name}"')
        HanaTestUtils.execute_sql(conn, f'SET SCHEMA "{schema_name}"')

    @staticmethod
    def drop_table(conn: dbapi.Connection, table_name: str) -> None:
        cur = conn.cursor()
        try:
            cur.execute(f'DROP TABLE "{table_name}"')
        except dbapi.Error as e:
            raise RuntimeError(f"Error dropping table {table_name}: {e}")
        finally:
            cur.close()

    @staticmethod
    def execute_sparql_query(
        connection: dbapi.Connection, query: str, request_headers: str
    ) -> str:
        """Execute a SPARQL query."""
        cursor = connection.cursor()
        try:
            result = cursor.callproc(
                "SYS.SPARQL_EXECUTE", (query, request_headers, "?", "?")
            )
            response = result[2]
            return response
        except dbapi.Error as db_error:
            raise RuntimeError(
                f'The database query "{query}" failed: '
                f'{db_error.errortext.split("; Server Connection")[0]}'
            )
        finally:
            cursor.close()
