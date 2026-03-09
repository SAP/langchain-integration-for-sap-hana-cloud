import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

class SqlOperand:
    def __init__(self, value):
        the_type = type(value)

        # Handle plain values.
        if the_type is bool:
            self.the_type = "BOOLEAN"
            self.placeholder = "TO_BOOLEAN(?)"
            self.value = "true" if value else "false"
            return
        if the_type in (int, float):
            self.the_type = "DOUBLE"
            self.placeholder = "TO_DOUBLE(?)"
            self.value = float(value)
            return
        if the_type is str:
            self.the_type = "NVARCHAR"
            self.placeholder = "TO_NVARCHAR(?)"
            self.value = value
            return

        # Handle special values.
        if isinstance(value, dict) and ("type" in value):
            if value["type"] == "date":
                self.the_type = "DATE"
                self.placeholder = "TO_DATE(?)"
                self.value = value["date"]
                if self.value:
                    return
                # There should be no empty date, fall through to ValueError.
        
        # If we reach this point, the value type is not supported.
        raise ValueError(f"Cannot deduce SQL operand for {value}")

    def __str__(self):
        # We do not want to print internal types.
        # Users of langchain should see their input value in error messages.
        assert False

def _determine_sql_operands(operands: list[any]) -> list[SqlOperand]:
    if not isinstance(operands, list):
        raise ValueError(f"Expected list of operands, but got {operands}")
    return [SqlOperand(op) for op in operands]

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
            else:
                # Value represents a typed SQL value. Throws for illegal value.
                sql_operand = SqlOperand(value)
                ret_sql_clause = f"{self._create_selector(key)} = {sql_operand.placeholder}"
                ret_query_tuple = [sql_operand.value]
            statements.append(ret_sql_clause)
            parameters += ret_query_tuple
        return _sql_serialize_logical_clauses("AND", statements), parameters

    def _sql_serialize_column_operation(
        self, column: str, operator: str, operands: any
    ) -> Tuple[str, List]:
        if operator == "$contains":
            sql_operand = SqlOperand(operands)
            if not isinstance(operands, str):
                raise ValueError(f"Expected a string operand for $contains, but got {operands}")
            if not operands:
                raise ValueError(f"Expected a non-empty string operand for $contains")
            statement = (
                f"SCORE({sql_operand.placeholder} IN (\"{column}\" EXACT SEARCH MODE 'text')) > 0"
            )
            return statement, [sql_operand.value]
        selector = self._create_selector(column)
        if operator == "$like":
            sql_operand = SqlOperand(operands)
            if sql_operand.the_type != "NVARCHAR":
                raise ValueError(f"Expected a string operand for $like, but got {operands}")
            statement = f"{selector} LIKE {sql_operand.placeholder}"
            return statement, [sql_operand.value]
        if operator == "$between":
            sql_operands = _determine_sql_operands(operands)
            if len(operands) != 2:
                raise ValueError(f"Expected 2 operands for $between, but got {len(operands)}")
            if type(operands[0]) is not type(operands[1]):
                raise ValueError(f"Expected operands of the same type for $between, but got {operands}")
            sql_from, sql_to = sql_operands
            if sql_from.the_type not in ("DOUBLE", "NVARCHAR", "DATE"):
                raise ValueError(f"Expected operand types (int, float, str, date) for $between, but got {operands[0]}")
            statement = (
                f"{selector} BETWEEN {sql_from.placeholder} AND {sql_to.placeholder}"
            )
            return statement, [sql_from.value, sql_to.value]
        if operator in ("$in", "$nin"):
            sql_operator = {
                "$in": "IN",
                "$nin": "NOT IN",
            }[operator]
            sql_operands = _determine_sql_operands(operands)
            if len(operands) == 0:
                raise ValueError(f"Expected a non-empty list of operands for {operator}")
            for operand in operands:
                if type(operand) is not type(operands[0]):
                    raise ValueError(f"Expected operands of the same type for {operator}, but got {operands}")
            sql_placeholders = [sql_operand.placeholder for sql_operand in sql_operands]
            sql_values = [sql_operand.value for sql_operand in sql_operands]
            statement = f"{selector} {sql_operator} ({', '.join(sql_placeholders)})"
            return statement, sql_values
        if operator in ("$eq", "$ne"):
            # Allow null checks for equality operators.
            if operands is None:
                sql_operation = {
                    "$eq": "IS NULL",
                    "$ne": "IS NOT NULL",
                }[operator]
                statement = f"{selector} {sql_operation}"
                return statement, []
            sql_operator = {
                "$eq": "=",
                "$ne": "<>",
            }[operator]
            sql_operand = SqlOperand(operands)
            statement = f"{selector} {sql_operator} {sql_operand.placeholder}"
            return statement, [sql_operand.value]
        if operator in ("$gt", "$gte", "$lt", "$lte"):
            sql_operator = {
                "$gt": ">",
                "$gte": ">=",
                "$lt": "<",
                "$lte": "<=",
            }[operator]
            sql_operand = SqlOperand(operands)
            if sql_operand.the_type not in ("DOUBLE", "NVARCHAR", "DATE"):
                raise ValueError(f"Expected operand types (int, float, str, date) for {operator}, but got {operands}")
            statement = f"{selector} {sql_operator} {sql_operand.placeholder}"
            return statement, [sql_operand.value]
        
        # Unknown operation if we reach this point.
        raise ValueError(f"Unsupported column operation for {operator=}, {operands=}")

    def _sql_serialize_logical_operation(
        self, operator: str, operands: List
    ) -> Tuple[str, List]:
    
        if not isinstance(operands, list):
            raise ValueError(f"Expected a list of operands for {operator=}, but got {operands}")
        if len(operands) < 2:
            raise ValueError(f"Expected a list of at least 2 operands for {operator=}, but got {len(operands)}")
        if operator in ("$and", "$or"):
            sql_clauses, query_tuple = [], []
            for operand in operands:
                ret_sql_clause, ret_query_tuple = self._create_where_clause(operand)
                sql_clauses.append(ret_sql_clause)
                query_tuple += ret_query_tuple
                sql_operator = {
                    "$and": "AND",
                    "$or": "OR",
                }[operator]
            return _sql_serialize_logical_clauses(sql_operator, sql_clauses), query_tuple
        
        # If we reach this point, the operator is not supported.
        raise ValueError(f"Unsupported logical operation for {operator=}, {operands=}")

    def _create_selector(self, column: str) -> str:
        if column in self.specific_metadata_columns:
            return f'"{column}"'
        else:
            return f"JSON_VALUE({self.metadata_column}, '$.{column}')"
