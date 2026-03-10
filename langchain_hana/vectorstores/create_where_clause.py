import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


class FilterOperand:
    """Represents a filter operand with type information for validation and error messages."""
    
    def __init__(self, value):
        if isinstance(value, (bool, int, float, str)):
            self.value = value
            self.the_type = type(value).__name__
        elif isinstance(value, dict) and value.get("type") == "date":
            if "date" not in value:
                raise ValueError(f"Date operand missing 'date' key: {value!r}")
            self.value = value["date"]
            if not self.value:
                raise ValueError("Date operand with empty value")
            self.the_type = "date"
        else:
            raise ValueError(f"Operand cannot be created from {value!r}")
    
    def __str__(self) -> str:
        return f"{self.value!r} ({self.the_type})"
    
    def __repr__(self) -> str:
        return str(self)


class SqlOperand:
    """SQL operand with placeholder and value for parameterized queries."""
    
    def __init__(self, operand: FilterOperand):
        """Construct SqlOperand from a FilterOperand."""
        if operand.the_type == "bool":
            self.the_type = "BOOLEAN"
            self.placeholder = "TO_BOOLEAN(?)"
            self.value = "true" if operand.value else "false"
        elif operand.the_type in ("int", "float"):
            self.the_type = "DOUBLE"
            self.placeholder = "TO_DOUBLE(?)"
            self.value = float(operand.value)
        elif operand.the_type == "str":
            self.the_type = "NVARCHAR"
            self.placeholder = "TO_NVARCHAR(?)"
            self.value = operand.value
        elif operand.the_type == "date":
            self.the_type = "DATE"
            self.placeholder = "TO_DATE(?)"
            self.value = operand.value
        else:
            # This should not happen if FilterOperand is constructed correctly.
            raise AssertionError(f"Unreachable. {operand=}")

    def __str__(self):
        # We do not want to print internal types.
        # Users of langchain should see their input value in error messages.
        assert False


def _determine_filter_operands(operator: str, operands: any) -> list[FilterOperand]:
    """Check that operands is a list and return list of FilterOperands."""
    if not isinstance(operands, (list, tuple)):
        raise ValueError(f"Operator {operator} expects list/tuple of operands, but got {operands}")
    if len(operands) == 0:
        raise ValueError(f"Operator {operator} expects at least 1 operand")
    return [_determine_single_filter_operand(operator, op) for op in operands]


def _determine_single_filter_operand(operator: str, operands: any) -> FilterOperand:
    """Check that operands is a single value (not list/tuple) and return FilterOperand."""
    if isinstance(operands, (list, tuple)):
        raise ValueError(
            f"Operator {operator} expects a single operand, but got {type(operands).__name__}: {operands}"
        )
    try:
        return FilterOperand(operands)
    except ValueError as e:
        error_message = str(e)
        raise ValueError(f"Operator {operator}: {error_message}")


def _determine_sql_operands(operator: str, operands: any) -> list[SqlOperand]:
    """Check that operands is a list and return list of SqlOperands."""
    filter_operands = _determine_filter_operands(operator, operands)
    return [SqlOperand(op) for op in filter_operands]

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
                        f"Filter expects a single 'operator: operands' entry, but got {value}"
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
                # Value represents a typed SQL value (implicit $eq operator).
                try:
                    operand = FilterOperand(value)
                except ValueError:
                    raise ValueError(
                        f"Implicit operator $eq received unsupported operand: {value!r}"
                    )
                sql_operand = SqlOperand(operand)
                ret_sql_clause = f"{self._create_selector(key)} = {sql_operand.placeholder}"
                ret_query_tuple = [sql_operand.value]
            statements.append(ret_sql_clause)
            parameters += ret_query_tuple
        return _sql_serialize_logical_clauses("AND", statements), parameters

    def _sql_serialize_column_operation(
        self, column: str, operator: str, operands: any
    ) -> Tuple[str, List]:
        if operator == "$contains":
            operand = _determine_single_filter_operand(operator, operands)
            if operand.the_type != "str" or not operand.value:
                raise ValueError(f"Operator $contains expects a non-empy string operand, but got {operand!r}")
            sql_operand = SqlOperand(operand)
            statement = (
                f"SCORE({sql_operand.placeholder} IN (\"{column}\" EXACT SEARCH MODE 'text')) > 0"
            )
            return statement, [sql_operand.value]
        selector = self._create_selector(column)
        if operator == "$like":
            operand = _determine_single_filter_operand(operator, operands)
            if operand.the_type != "str":
                raise ValueError(f"Operator $like expects a string operand, but got {operand}")
            sql_operand = SqlOperand(operand)
            statement = f"{selector} LIKE {sql_operand.placeholder}"
            return statement, [sql_operand.value]
        if operator == "$between":
            filter_operands = _determine_filter_operands(operator, operands)
            if len(filter_operands) != 2:
                raise ValueError(f"Operator $between expects 2 operands, but got {filter_operands}")
            from_operand, to_operand = filter_operands
            if from_operand.the_type != to_operand.the_type:
                raise ValueError(f"Operator $between expects operands of the same type, but got {filter_operands}")
            if from_operand.the_type not in ("int", "float", "str", "date"):
                raise ValueError(f"Operator $between expects operand types (int, float, str, date), but got {filter_operands}")
            sql_from = SqlOperand(from_operand)
            sql_to = SqlOperand(to_operand)
            statement = (
                f"{selector} BETWEEN {sql_from.placeholder} AND {sql_to.placeholder}"
            )
            return statement, [sql_from.value, sql_to.value]
        if operator in ("$in", "$nin"):
            sql_operator = {
                "$in": "IN",
                "$nin": "NOT IN",
            }[operator]
            filter_operands = _determine_filter_operands(operator, operands)
            for op in filter_operands:
                if op.the_type != filter_operands[0].the_type:
                    raise ValueError(f"Operator {operator} expects operands of the same type, but got {operands}")
            sql_operands = [SqlOperand(op) for op in filter_operands]
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
            operand = _determine_single_filter_operand(operator, operands)
            sql_operand = SqlOperand(operand)
            statement = f"{selector} {sql_operator} {sql_operand.placeholder}"
            return statement, [sql_operand.value]
        if operator in ("$gt", "$gte", "$lt", "$lte"):
            operand = _determine_single_filter_operand(operator, operands)
            
            # Check if the operand type is allowed for comparison operators.
            if operand.the_type not in ("int", "float", "str", "date"):
                raise ValueError(
                    f"Operator {operator} expects operand of type int/float/str/date, but got {operand}"
                )
            
            sql_operator = {
                "$gt": ">",
                "$gte": ">=",
                "$lt": "<",
                "$lte": "<=",
            }[operator]
            sql_operand = SqlOperand(operand)
            statement = f"{selector} {sql_operator} {sql_operand.placeholder}"
            return statement, [sql_operand.value]
        
        # Unknown operation if we reach this point.
        raise ValueError(f"Operator {operator} is not supported")

    def _sql_serialize_logical_operation(
        self, operator: str, operands: List
    ) -> Tuple[str, List]:
    
        if not isinstance(operands, list):
            raise ValueError(f"Operator {operator} expects a list of operands, but got {operands!r}")
        if len(operands) < 2:
            raise ValueError(f"Operator {operator} expects at least 2 operands, but got {operands!r}")
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
        raise ValueError(f"Operator {operator} is not supported")

    def _create_selector(self, column: str) -> str:
        if column in self.specific_metadata_columns:
            return f'"{column}"'
        else:
            return f"JSON_VALUE({self.metadata_column}, '$.{column}')"
