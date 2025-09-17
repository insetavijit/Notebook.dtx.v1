# Enterprise-Level Technical Analysis Framework Documentation

## Overview

The Enterprise-Level Technical Analysis Framework is a Python-based library designed for performing technical analysis on financial data. It provides a fluent interface for conducting comparisons and detections (e.g., crossovers, thresholds) on pandas DataFrames, with built-in support for error handling, logging, and extensibility. The framework is particularly suited for enterprise environments where robustness, maintainability, and clear audit trails of operations are critical.

Key goals:
- Enable chainable operations for intuitive analysis workflows.
- Ensure comprehensive validation and exception handling to prevent runtime errors in production.
- Support natural language query parsing for non-programmatic users.
- Maintain an operations log for auditing and debugging.

This documentation is structured for use in Obsidian, with sections as top-level headings for easy linking and navigation. Code snippets are formatted for syntax highlighting.

## Features

- **Fluent API**: Chain methods like `.above()`, `.below()`, `.crossed_up()` for seamless analysis.
- **Comparison Operations**: Supports above/below thresholds, crossover detections (up/down).
- **Natural Language Query Support**: Parse multi-line queries into operations.
- **Validation and Error Handling**: Strict column and type checks with custom exceptions.
- **Logging**: Integrated logging for operations tracking.
- **Extensibility**: Register custom comparators via factory.
- **Results Management**: Access signals, active signals, summaries, and reset state.
- **Dependencies**: Built on pandas, numpy, mplfinance (for potential charting extensions), and standard Python libraries.

## Installation and Requirements

### Prerequisites
- Python 3.8+ (tested up to 3.12).
- Required packages:
  - `pandas` (>=2.0)
  - `numpy` (>=1.24)
  - `mplfinance` (>=0.12, optional for visualization extensions)
- No external services or databases required.

### Installation
1. Copy the provided code into a file, e.g., `technical_analyzer.py`.
2. Install dependencies via pip:
   ```
   pip install pandas numpy mplfinance
   ```
3. Import and use in your scripts:
   ```python
   from technical_analyzer import cabr  # Factory function
   ```

For enterprise deployment:
- Package as a Python module or wheel.
- Use virtual environments (e.g., venv or conda) for isolation.
- Integrate with CI/CD pipelines for testing.

## Architecture

The framework follows a modular, object-oriented design:

### Core Components
- **Validators (`ColumnValidator`)**: Static methods for checking column existence, numeric types, and value conversions.
- **Comparators (`BaseComparator` and subclasses)**: Abstract base for operations like `AboveComparator`, `BelowComparator`, etc. Each handles a specific comparison logic.
- **Factory (`ComparatorFactory`)**: Creates comparator instances; supports registration of custom comparators.
- **Parser (`QueryParser`)**: Converts natural language strings into structured operation dictionaries.
- **Analyzer (`TechnicalAnalyzer`)**: Central class managing the DataFrame, executing operations, and logging results.
- **Enums and Data Classes**: `ComparisonType` for operations; `AnalysisResult` for encapsulating outcomes.
- **Exceptions**: Custom `TAException` for framework-specific errors.

### Data Flow
1. Initialize with a pandas DataFrame.
2. Chain methods or execute queries.
3. Each operation:
   - Validates inputs.
   - Applies comparison.
   - Adds a new boolean/int column.
   - Logs the result.
4. Retrieve summaries, signals, or reset.

### Extensibility Points
- Inherit from `BaseComparator` and register via `ComparatorFactory.register_comparator()`.
- Override parsing logic in `QueryParser` for custom query formats.

## API Reference

### Enums
#### `ComparisonType`
Enumeration of supported operations:
- `ABOVE = "above"`
- `BELOW = "below"`
- `CROSSED_UP = "crossed_up"`
- `CROSSED_DOWN = "crossed_dn"`
- `EQUALS = "equals"` (not implemented in base, extensible)
- `GREATER_EQUAL = "greater_equal"` (extensible)
- `LESS_EQUAL = "less_equal"` (extensible)

### Data Classes
#### `AnalysisResult`
Encapsulates operation outcomes:
- `column_name: str` - Name of the generated column.
- `operation: str` - Description of the operation.
- `success: bool` - True if successful.
- `message: str = ""` - Optional message or error.
- `data: Optional[pd.Series] = None` - Resulting series.

### Classes

#### `ColumnValidator`
Static utility for validations:
- `validate_column_exists(df: pd.DataFrame, column: str) -> bool`
  - Raises `TAException` if column missing.
- `validate_numeric_column(df: pd.DataFrame, column: str) -> bool`
  - Ensures column is numeric.
- `validate_numeric_value(value: Union[str, int, float]) -> float`
  - Converts to float or raises exception.

#### `BaseComparator` (Abstract)
Base for all comparators:
- `compare(df: pd.DataFrame, x: str, y: Union[str, float], new_col: Optional[str] = None) -> pd.DataFrame`
  - Abstract method to implement comparison.
- `_generate_column_name(x: str, y: Union[str, float], operation: str) -> str`
  - Creates names like `Close_above_EMA_21`.
- `_add_constant_column(df: pd.DataFrame, name: str, value: float) -> pd.DataFrame`
  - Adds constant columns for numeric thresholds.

#### Concrete Comparators
- `AboveComparator`: `df[new_col] = (df[x] > df[y]).astype(int)`
- `BelowComparator`: `df[new_col] = (df[x] < df[y]).astype(int)`
- `CrossedUpComparator`: Detects upward cross using shifted diff.
- `CrossedDownComparator`: Detects downward cross using shifted diff.

#### `ComparatorFactory`
- `get_comparator(operation: str) -> BaseComparator`
  - Returns instance or raises exception.
- `register_comparator(operation: str, comparator: BaseComparator)`
  - Adds custom comparators.

#### `QueryParser`
- `parse_query(query: str) -> List[Dict[str, str]]`
  - Splits lines into {'column1', 'operation', 'column2'} dicts.
  - Skips malformed lines with warnings.

#### `TechnicalAnalyzer`
Main interface:
- `__init__(self, df: pd.DataFrame)`
  - Copies DataFrame; logs shape.
- Properties:
  - `df: pd.DataFrame` - Current DataFrame.
  - `operations_log: List[AnalysisResult]` - All results.
- Fluent Methods:
  - `above(x: str, y: Union[str, float], new_col: Optional[str] = None) -> TechnicalAnalyzer`
  - `below(...)` (similar)
  - `crossed_up(...)`
  - `crossed_down(...)`
- `execute_query(query: str) -> TechnicalAnalyzer`
  - Parses and executes multi-line queries.
- `get_signals(column: str) -> pd.Series`
  - Returns the signal series.
- `get_active_signals(column: str) -> pd.DataFrame`
  - Filters rows where signal == 1.
- `summary() -> pd.DataFrame`
  - Tabular summary of operations, including active signal counts.
- `reset() -> TechnicalAnalyzer`
  - Removes generated columns; clears log.

### Factory Function
- `cabr(df: pd.DataFrame) -> TechnicalAnalyzer`
  - Convenience creator (acronym for "Create Analyzer By Reference").

### Exceptions
- `TAException(Exception)`
  - Raised for validation failures, unsupported operations, etc.

### Logging
- Configured at INFO level.
- Logs operations success/failure, DataFrame initialization, resets.

## Usage Examples

### Basic Fluent Chain
```python
import pandas as pd
import numpy as np
from technical_analyzer import cabr

# Sample Data
dates = pd.date_range('2023-01-01', periods=100)
df = pd.DataFrame({
    'DateTime': dates,
    'Close': 100 + np.cumsum(np.random.randn(100)),
    'EMA_21': 100 + np.cumsum(np.random.randn(100) * 0.5),
})

analyzer = cabr(df)
analyzer.above('Close', 'EMA_21').below('Close', 105)

print(analyzer.summary())
print(analyzer.df.head())
```

### Natural Language Query
```python
query = """
Close above EMA_21
Close crossed_up EMA_21
"""
analyzer.execute_query(query)
active = analyzer.get_active_signals('Close_crossed_up_EMA_21')
print(active)
```

### Custom Comparator Example
```python
class EqualsComparator(BaseComparator):
    def compare(self, df, x, y, new_col=None):
        # Implementation...
        pass

ComparatorFactory.register_comparator('equals', EqualsComparator())
analyzer._execute_operation('equals', 'Close', 100)
```

### Resetting State
```python
analyzer.reset()
print(analyzer.df.columns)  # Only original columns
```

## Integration with Visualization
While not core, integrate with mplfinance:
```python
import mplfinance as mpf
add_plots = [mpf.make_addplot(analyzer.df['Close_above_EMA_21'], type='scatter')]
mpf.plot(analyzer.df.set_index('DateTime'), type='candle', addplot=add_plots)
```

## Best Practices
- Always validate input DataFrames for required columns (e.g., 'Close', 'EMA').
- Use try-except blocks around chains for production resilience.
- Limit chain length for readability; use queries for complex logic.
- Monitor logs in enterprise logging systems (e.g., integrate with ELK stack).
- Test with edge cases: empty DataFrames, non-numeric columns, NaNs.
- For large datasets, consider performance optimizations (e.g., vectorized operations are already used).

## Error Handling and Troubleshooting
- **Common Errors**:
  - `TAException: Column 'X' not found`: Ensure column exists.
  - `TAException: Unsupported operation`: Check spelling; extend if needed.
  - Numeric conversion failures: Clean data beforehand.
- **Debugging**:
  - Inspect `operations_log` for failures.
  - Use `summary()` post-execution.
  - Enable DEBUG logging for verbose output.
- **Known Limitations**:
  - No built-in NaN handling in comparisons (use `df.fillna()` prior).
  - Query parser assumes simple "Col op Value" format; no complex logic.
  - Not thread-safe; use in single-threaded contexts.

## Version History
- v1.0 (Initial): Base framework with core comparators.
- Future: Add more operations (e.g., equals, divergences); integrate TA-Lib.

For contributions or issues, refer to the code repository (if applicable). This documentation is self-contained as of September 17, 2025.
