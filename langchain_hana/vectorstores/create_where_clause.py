import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

COMPARISONS_TO_SQL = {
    "$eq": "=",
    "$ne": "<>",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

IN_OPERATORS_TO_SQL = {
    "$in": "IN",
    "$nin": "NOT IN",
}

BETWEEN_OPERATOR = "$between"

LIKE_OPERATOR = "$like"

CONTAINS_OPERATOR = "$contains"


class ContainsNeedsSpecialSqlSyntax:
    def __repr__(self) -> None:  # type: ignore[override]
        raise AssertionError(f"{CONTAINS_OPERATOR} needs special SQL syntax")


COLUMN_OPERATORS = {
    **COMPARISONS_TO_SQL,
    **IN_OPERATORS_TO_SQL,
    BETWEEN_OPERATOR: "BETWEEN",
    LIKE_OPERATOR: "LIKE",
    CONTAINS_OPERATOR: ContainsNeedsSpecialSqlSyntax(),
}

LOGICAL_OPERATORS_TO_SQL = {"$and": "AND", "$or": "OR"}

class CreateWhereClause:
    def __init__(self, hanaDb: Any) -> None:
        self.specific_metadata_columns = hanaDb.specific_metadata_columns
        self.metadata_column = hanaDb.metadata_column

    def __call__(self, filter):  # type: ignore[no-untyped-def]
        """Serializes filter to a where clause (prepared_statement) and parameters

        The where clause should be appended to an existing SQL statement.

        Example usage:
        where_clause, parameters = CreateWhereClause(hanaDb)(filter)
        cursor.execute(f"{stmt} {where_clause}", parameters)
        """
        if filter:
            statement, parameters = self._create_where_clause(filter)
            assert statement.count("?") == len(parameters)
            return f"WHERE {statement}", parameters
        else:
            return "", []

    def _create_where_clause(self, filter: dict) -> Tuple[str, List]:
        if not filter:
            raise ValueError("Empty filter")
        statements = []
        parameters = []
        for key, value in filter.items():
            if key.startswith("$"):
                # Generic filter objects may only have logical operators.
                if key not in LOGICAL_OPERATORS_TO_SQL:
                    raise ValueError(f"Unexpected operator {key=} in {filter=}")
                ret_sql_clause, ret_query_tuple = self._sql_serialize_logical_operation(
                    key, value
                )
            else:
                if not isinstance(value, (bool, int, str, dict)):
                    raise ValueError(f"Unsupported filter value type: {type(value)}")
                if isinstance(value, dict) and "type" not in value:
                    # Value is an operator.
                    if len(value) != 1:
                        raise ValueError(
                            "Expecting a single entry 'operator: operands'"
                            f", but got {value=}"
                        )
                    operator, operands = list(value.items())[0]
                    ret_sql_clause, ret_query_tuple = (
                        self._sql_serialize_column_operation(key, operator, operands)
                    )
                else:
                    placeholder, value = (
                        CreateWhereClause._determine_typed_sql_placeholder(value)
                    )
                    ret_sql_clause = f"{self._create_selector(key)} = {placeholder}"
                    ret_query_tuple = [value]
            statements.append(ret_sql_clause)
            parameters += ret_query_tuple
        return CreateWhereClause._sql_serialize_logical_clauses(
            "AND", statements
        ), parameters

    def _sql_serialize_column_operation(
        self, column: str, operator: str, operands: any
    ) -> Tuple[str, List]:
        if operator in LOGICAL_OPERATORS_TO_SQL:
            raise ValueError(
                f"Did not expect oerator from {LOGICAL_OPERATORS_TO_SQL=}"
                f", but got {operator=}"
            )
        if operator not in COLUMN_OPERATORS:
            raise ValueError(f"{operator=} not in {COLUMN_OPERATORS.keys()=}")
        if operator == CONTAINS_OPERATOR:
            placeholder, value = CreateWhereClause._determine_typed_sql_placeholder(
                operands
            )
            statement = (
                f"SCORE({placeholder} IN (\"{column}\" EXACT SEARCH MODE 'text')) > 0"
            )
            return statement, [value]
        sql_operator = COLUMN_OPERATORS[operator]
        selector = self._create_selector(column)
        if operator == BETWEEN_OPERATOR:
            if len(operands) != 2:
                raise ValueError(f"Expected 2 operands, but got {operands=}")
            from_placeholder, from_value = (
                CreateWhereClause._determine_typed_sql_placeholder(operands[0])
            )
            to_placeholder, to_value = (
                CreateWhereClause._determine_typed_sql_placeholder(operands[1])
            )
            statement = (
                f"{selector} {sql_operator} {from_placeholder} AND {to_placeholder}"
            )
            return statement, [from_value, to_value]
        if operator in IN_OPERATORS_TO_SQL:
            if not isinstance(operands, list):
                raise ValueError(f"Expected a list, but got {operands=}")
            placeholder_value_list = [
                CreateWhereClause._determine_typed_sql_placeholder(item)
                for item in operands
            ]
            placeholders = ", ".join([item[0] for item in placeholder_value_list])
            values = [item[1] for item in placeholder_value_list]
            statement = f"{selector} {sql_operator} ({placeholders})"
            return statement, values
        # Default behavior for single value operators.
        placeholder, value = CreateWhereClause._determine_typed_sql_placeholder(
            operands
        )
        statement = f"{selector} {sql_operator} {placeholder}"
        return statement, [value]

    @staticmethod
    def _determine_typed_sql_placeholder(value):  # type: ignore[no-untyped-def]

        the_type = type(value)

        # Handle plain values.
        if the_type is bool:
            return "TO_BOOLEAN(?)", "true" if value else "false"
        if the_type in (int, float):
            return "TO_DOUBLE(?)", value
        
        # Do not accept empty values.
        if not value:
            raise ValueError("No operands provided")

        # Handle container types: only allowed for dates.
        if isinstance(value, dict) and ("type" in value) and (value["type"] == "date"):
            return "TO_DATE(?)", value["date"]
        if isinstance(value, (dict, list, tuple)):
            raise ValueError(f"Cannot handle {value=}")
        
        logger.warning(f"Plain SQL Placeholder '?' for {value=}")
        return "?", value

    @staticmethod
    def _sql_serialize_logical_clauses(
        sql_operator: str, sql_clauses: list[str]
    ) -> str:
        supported_operators = LOGICAL_OPERATORS_TO_SQL.values()
        if sql_operator not in supported_operators:
            raise ValueError(f"{sql_operator=}, is not in {supported_operators=}")
        if not sql_clauses:
            raise ValueError("sql_clauses is empty")
        if not all(sql_clauses):
            raise ValueError(f"Empty sql clause in {sql_clauses=}")
        if len(sql_clauses) == 1:
            return sql_clauses[0]
        return f" {sql_operator} ".join([f"({clause})" for clause in sql_clauses])

    def _sql_serialize_logical_operation(
        self, operator: str, operands: List
    ) -> Tuple[str, List]:
        if not isinstance(operands, list):
            raise ValueError(f"Unexpected operands for {operator=}: {operands=}")
        if operator not in LOGICAL_OPERATORS_TO_SQL:
            raise ValueError(
                f"Expected operator from {LOGICAL_OPERATORS_TO_SQL=}"
                f", but got {operator=}"
            )
        sql_clauses, query_tuple = [], []
        for operand in operands:
            ret_sql_clause, ret_query_tuple = self._create_where_clause(operand)
            sql_clauses.append(ret_sql_clause)
            query_tuple += ret_query_tuple
        return (
            CreateWhereClause._sql_serialize_logical_clauses(
                LOGICAL_OPERATORS_TO_SQL[operator], sql_clauses
            ),
            query_tuple,
        )

    def _create_selector(self, column: str) -> str:
        if column in self.specific_metadata_columns:
            return f'"{column}"'
        else:
            return f"JSON_VALUE({self.metadata_column}, '$.{column}')"
