import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

def is_date_value(value: Any) -> bool:
    return isinstance(value, dict) and ("type" in value) and (value["type"] == "date")

def _determine_typed_sql_placeholder(value):  # type: ignore[no-untyped-def]

        the_type = type(value)

        # Handle plain values.
        if the_type is bool:
            return "TO_BOOLEAN(?)", "true" if value else "false"
        if the_type in (int, float):
            return "TO_DOUBLE(?)", value
        if the_type is str:
            return "TO_NVARCHAR(?)", value

        # Handle container types: only allowed for dates.
        if is_date_value(value):
            return "TO_DATE(?)", value["date"]
        
        # If we reach this point, the value type is not supported.
        raise ValueError(f"Unsupported filter value type: {the_type}, value: {value}")

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
                ret_sql_clause, ret_query_tuple = self._sql_serialize_logical_operation(
                    key, value
                )
            elif isinstance(value, dict) and "type" not in value:
                # Value is a column operator.
                if len(value) != 1:
                    raise ValueError(
                        "Expecting a single entry 'operator: operands'"
                        f", but got {value=}"
                    )
                operator, operands = list(value.items())[0]
                ret_sql_clause, ret_query_tuple = (
                    self._sql_serialize_column_operation(key, operator, operands)
                )
            elif value is None:
                # Value is plain NULL.
                ret_sql_clause = f"{self._create_selector(key)} IS NULL"
                ret_query_tuple = []
            elif is_date_value(value) or isinstance(value, (int, float, str, bool)):
                # Value represents a typed SQL value.
                # _determine_typed_sql_placeholder throws for illegal types.
                placeholder, value = (
                    _determine_typed_sql_placeholder(value)
                )
                ret_sql_clause = f"{self._create_selector(key)} = {placeholder}"
                ret_query_tuple = [value]
            else:
                raise ValueError(
                    f"Invalid filter value with {key=}, {value=}"
                )
            statements.append(ret_sql_clause)
            parameters += ret_query_tuple
        return CreateWhereClause._sql_serialize_logical_clauses(
            "AND", statements
        ), parameters

    def _sql_serialize_column_operation(
        self, column: str, operator: str, operands: any
    ) -> Tuple[str, List]:
        if operator == "$contains":
            if not isinstance(operands, str) or not operands:
                raise ValueError(f"Expected a non-empty string operand for {operator=}, but got {operands=}")
            sql_placeholder, sql_value = _determine_typed_sql_placeholder(
                operands
            )
            statement = (
                f"SCORE({sql_placeholder} IN (\"{column}\" EXACT SEARCH MODE 'text')) > 0"
            )
            return statement, [sql_value]
        selector = self._create_selector(column)
        if operator == "$like":
            if not isinstance(operands, str):
                raise ValueError(f"Expected a string operand for {operator=}, but got {operands=}")
            sql_placeholder, sql_value = _determine_typed_sql_placeholder(
                operands
            )
            statement = f"{selector} LIKE {sql_placeholder}"
            return statement, [sql_value]
        if operator == "$between":
            if not isinstance(operands, list) or len(operands) != 2:
                raise ValueError(f"Expected a list of two operands for {operator=}, but got {operands=}")
            if type(operands[0]) != type(operands[1]):
                raise ValueError(f"Expected operands of the same type for {operator=}, but got {operands=}")
            if isinstance(operands[0], bool) or not (isinstance(operands[0], (int, float, str)) or is_date_value(operands[0])):
                raise ValueError(f"Expected a list of (int, float, str, date) for {operator=}, but got {operands=}")
            from_sql_placeholder, from_sql_value = (
                _determine_typed_sql_placeholder(operands[0])
            )
            to_sql_placeholder, to_sql_value = (
                _determine_typed_sql_placeholder(operands[1])
            )
            statement = (
                f"{selector} BETWEEN {from_sql_placeholder} AND {to_sql_placeholder}"
            )
            return statement, [from_sql_value, to_sql_value]
        if operator in ("$in", "$nin"):
            if not isinstance(operands, list) or len(operands) == 0:
                raise ValueError(f"Expected a non-empty list of operands for {operator=}, but got {operands=}")
            check_type = {type(operand) for operand in operands}
            if len(check_type) > 1:
                raise ValueError(f"Expected operands of the same type for {operator=}, but got {operands=}")
            if not (list(check_type)[0] in (int, float, str, bool) or all(is_date_value(operand) for operand in operands)):
                raise ValueError(f"Expected a list of (int, float, str, bool, date) for {operator=}, but got {operands=}")
            sql_placeholder_value_list = [
                _determine_typed_sql_placeholder(item)
                for item in operands
            ]
            if operator == "$in":
                sql_operator = "IN"
            if operator == "$nin":
                sql_operator = "NOT IN"
            placeholders = ", ".join([item[0] for item in sql_placeholder_value_list])
            sql_values = [item[1] for item in sql_placeholder_value_list]
            statement = f"{selector} {sql_operator} ({placeholders})"
            return statement, sql_values
        if operator in ("$eq", "$ne"):
            if not (isinstance(operands, (int, float, str, bool)) or is_date_value(operands) or operands is None):
                raise ValueError(f"Expected a (int, float, str, bool, date, None) for {operator=}, but got {operands=}")
            # Allow null checks for equality operators.
            if operands is None:
                if operator == "$eq":
                    sql_operator = "IS NULL"
                if operator == "$ne":
                    sql_operator = "IS NOT NULL"
                statement = f"{selector} {sql_operator}"
                return statement, []
            sql_operator = "=" if operator == "$eq" else "<>"
            sql_placeholder, sql_value = _determine_typed_sql_placeholder(operands)
            statement = f"{selector} {sql_operator} {sql_placeholder}"
            return statement, [sql_value]
        if operator in ("$gt", "$gte", "$lt", "$lte"):
            if  isinstance(operands, bool) or not (isinstance(operands, (int, float, str)) or is_date_value(operands)):
                raise ValueError(f"Expected a (int, float, str, date) for {operator=}, but got {operands=}")
            comparisons_to_sql = {
                "$gt": ">",
                "$gte": ">=",
                "$lt": "<",
                "$lte": "<=",
            }
            sql_operator = comparisons_to_sql[operator]
            sql_placeholder, sql_value = _determine_typed_sql_placeholder(operands)
            statement = f"{selector} {sql_operator} {sql_placeholder}"
            return statement, [sql_value]
        
        # Unknown operation if we reach this point.
        raise ValueError(f"Unsupported column operation for {operator=}, {operands=}")

    @staticmethod
    def _sql_serialize_logical_clauses(
        sql_operator: str, sql_clauses: list[str]
    ) -> str:
        if sql_operator not in ("AND", "OR"):
            raise ValueError(f"{sql_operator=}, is not in ('AND', 'OR')")
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
    
        if not isinstance(operands, list) or len(operands) < 2:
            raise ValueError(f"Expected a list of atleast two operands for {operator=}, but got {operands=}")
        if operator in ("$and", "$or"):
            sql_clauses, query_tuple = [], []
            for operand in operands:
                ret_sql_clause, ret_query_tuple = self._create_where_clause(operand)
                sql_clauses.append(ret_sql_clause)
                query_tuple += ret_query_tuple
                logical_operators_to_sql = {
                    "$and": "AND",
                    "$or": "OR",
                }
            return (
                CreateWhereClause._sql_serialize_logical_clauses(
                    logical_operators_to_sql[operator], sql_clauses
                ),
                query_tuple,
            )
        
        # If we reach this point, the operator is not supported.
        raise ValueError(f"Unsupported logical operation for {operator=}, {operands=}")

    def _create_selector(self, column: str) -> str:
        if column in self.specific_metadata_columns:
            return f'"{column}"'
        else:
            return f"JSON_VALUE({self.metadata_column}, '$.{column}')"
