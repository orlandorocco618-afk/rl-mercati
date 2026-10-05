"""
Robust Error Handler for Production Trading
================================================

This module provides error handling and logging functions
for every component of the RL-Mercati system.

Usage:
    from utils.error_handler import safe_execute, log_error, validate_data

    try:
        result = safe_execute(
            func=my_function,
            args=(arg1, arg2),
            error_context="Function/step name",
            critical=True
        )
    except Exception:
        pass  # Already logged
"""

import traceback
import sys
from functools import wraps
from typing import Any, Callable, Optional, Tuple


# ============================================================
#  CONSOLE COLORS AND FORMATTING
# ============================================================

class Colors:
    OK = "\033[92m"      # Green
    WARNING = "\033[93m" # Yellow
    ERROR = "\033[91m"   # Red
    INFO = "\033[94m"    # Blue
    RESET = "\033[0m"
    BOLD = "\033[1m"


# ============================================================
#  LOGGING ERRORS
# ============================================================

def log_error(
    error: Exception,
    context: str = "",
    critical: bool = False,
    show_traceback: bool = True
) -> str:
    """
    Logs an error in a standardized, human-readable way.

    Args:
        error: The exception
        context: Where the error happened (e.g. "load_csv" or "merge_h1_d1")
        critical: If True, it is printed as CRITICAL
        show_traceback: If True, prints the full traceback

    Returns:
        Formatted error message
    """
    level = f"{Colors.ERROR}[CRITICAL ERROR]{Colors.RESET}" if critical else f"{Colors.ERROR}[ERROR]{Colors.RESET}"

    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{level} in {Colors.BOLD}{context}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")

    print(f"\n{Colors.ERROR}Type: {type(error).__name__}{Colors.RESET}")
    print(f"{Colors.ERROR}Message: {str(error)}{Colors.RESET}")

    if show_traceback:
        print(f"\n{Colors.WARNING}Full traceback:{Colors.RESET}")
        print(traceback.format_exc())

    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    return f"[{context}] {type(error).__name__}: {str(error)}"


# ============================================================
#  SAFE EXECUTE
# ============================================================

def safe_execute(
    func: Callable,
    args: Tuple = (),
    kwargs: dict = None,
    error_context: str = "unknown",
    critical: bool = False,
    return_default: Any = None
) -> Any:
    """
    Runs a function with robust error handling.

    Args:
        func: Function to run
        args: Positional arguments
        kwargs: Keyword arguments
        error_context: Name/description of the operation
        critical: If True, the error is not handled (re-raised)
        return_default: Default value if the error is not critical

    Returns:
        The function result, or return_default on a non-critical error

    Raises:
        If critical=True
    """
    kwargs = kwargs or {}

    try:
        print(f"{Colors.INFO}[EXEC] {error_context}...{Colors.RESET}")
        result = func(*args, **kwargs)
        print(f"{Colors.OK}[OK] {error_context} completed{Colors.RESET}")
        return result

    except Exception as e:
        log_error(e, context=error_context, critical=critical, show_traceback=True)

        if critical:
            raise
        else:
            print(f"{Colors.WARNING}[FALLBACK] Continuing with the default value...{Colors.RESET}")
            return return_default


# ============================================================
#  VALIDATE DATA
# ============================================================

def validate_dataframe(
    df,
    expected_columns: list = None,
    min_rows: int = 100,
    context: str = "unknown",
    allow_nan: bool = False,
    critical: bool = False
) -> bool:
    """
    Validates a DataFrame to make sure it is usable.

    Args:
        df: DataFrame to validate
        expected_columns: Columns it MUST contain
        min_rows: Minimum number of rows
        context: Dataset name
        allow_nan: If False, it must NOT contain NaN

    Returns:
        True if valid, raises an exception otherwise
    """
    try:
        print(f"{Colors.INFO}[VALIDATE] {context}...{Colors.RESET}")

        # Quick-test mode: relax NaN checks to allow fast startup
        try:
            import os
            if os.getenv("RL_QUICK_TEST") == "1":
                allow_nan = True
        except Exception:
            pass

        # Check that it is a DataFrame
        try:
            import pandas as pd
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Not a DataFrame: {type(df)}")
        except Exception as e:
            raise ValueError(f"DataFrame check failed: {str(e)}")

        # Check rows
        if len(df) < min_rows:
            raise ValueError(f"Too few rows: {len(df)} < {min_rows}")

        # Check columns
        if expected_columns:
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

        # Check NaN
        if not allow_nan:
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                raise ValueError(f"DataFrame contains {nan_count} NaN values")

        print(f"{Colors.OK}[VALIDATE] ✓ {context} valid ({len(df)} rows){Colors.RESET}")
        return True

    except Exception as e:
        log_error(e, context=f"validate_dataframe({context})", critical=True)
        raise


# ============================================================
#  DECORATOR FOR AUTOMATIC ERROR HANDLING
# ============================================================

def handle_errors(error_context: str = "", critical: bool = False):
    """
    Decorator that adds automatic error handling to a function.

    Usage:
        @handle_errors(error_context="train_agent", critical=True)
        def train_agent():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = error_context or func.__name__
            try:
                print(f"{Colors.INFO}[RUNNING] {context}...{Colors.RESET}")
                result = func(*args, **kwargs)
                print(f"{Colors.OK}[COMPLETED] {context}{Colors.RESET}")
                return result
            except Exception as e:
                log_error(e, context=context, critical=critical)
                if critical:
                    raise
                return None
        return wrapper
    return decorator


# ============================================================
#  UTILS FOR COMMON ERRORS
# ============================================================




def check_file_exists(path: str, context: str = "file", critical: bool = False) -> bool:
    """Compatibility helper: checks whether a file exists, keeping the previous behaviour."""
    import os
    if not os.path.exists(path):
        error_msg = f"[CRITICAL ERROR] {context} NOT found: {path}"
        print(f"{Colors.ERROR}{error_msg}{Colors.RESET}")
        print(f"   Current folder: {os.getcwd()}")
        if critical:
            raise FileNotFoundError(error_msg)
        else:
            return False
    print(f"{Colors.OK}[OK] {context} found: {path}{Colors.RESET}")
    return True


def check_columns(df, required_cols: list, context: str = "data") -> bool:
    """Checks that a DataFrame has the required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        error_msg = f"[CRITICAL ERROR] Missing columns in {context}: {missing}"
        print(f"{Colors.ERROR}{error_msg}{Colors.RESET}")
        print(f"   Available columns: {list(df.columns)}")
        raise KeyError(error_msg)
    print(f"{Colors.OK}[OK] {context} has all the required columns{Colors.RESET}")
    return True


def check_datetime_column(df, col_name: str, context: str = "data") -> bool:
    """Checks that a column is in datetime format."""
    import pandas as pd
    if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
        error_msg = f"[CRITICAL ERROR] Column '{col_name}' is not datetime in {context}: {df[col_name].dtype}"
        print(f"{Colors.ERROR}{error_msg}{Colors.RESET}")
        raise TypeError(error_msg)
    print(f"{Colors.OK}[OK] Column '{col_name}' is datetime{Colors.RESET}")
    return True


# ============================================================
#  STATS & REPORTING
# ============================================================

def print_step_summary(
    step_name: str,
    status: str = "OK",  # "OK", "WARNING", "ERROR"
    details: dict = None,
    duration_sec: float = None,
    **kwargs
):
    """Prints a summary of a step with details.

    Accepts extra kwargs (e.g. rows=..) for compatibility with existing calls.
    """
    color = Colors.OK if status == "OK" else (Colors.WARNING if status == "WARNING" else Colors.ERROR)

    print(f"\n{Colors.BOLD}{color}[{status}] {step_name}{Colors.RESET}")

    merged = dict(details) if details else {}
    for k, v in kwargs.items():
        merged[k] = v

    if merged:
        for key, value in merged.items():
            print(f"   {key}: {value}")

    if duration_sec is not None:
        print(f"   Duration: {duration_sec:.2f}s")


def print_final_report(
    success: bool = None,
    status: str = None,
    total_time: float = None,
    stats: dict = None,
    errors: list = None
):
    """Prints a final report of the run.

    Compatibility: accepts `status` ("SUCCESS"/"FAILURE") or a boolean `success`.
    """
    if status is not None:
        success = True if str(status).upper().startswith("S") else False

    success = bool(success)
    status_color = Colors.OK if success else Colors.ERROR
    status_text = "✓ SUCCESS" if success else "✗ FAILURE"

    print(f"\n{Colors.BOLD}{status_color}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{status_color}{status_text}{Colors.RESET}")
    print(f"{Colors.BOLD}{status_color}{'='*70}{Colors.RESET}\n")

    if stats:
        print(f"{Colors.INFO}Statistics:{Colors.RESET}")
        for key, value in stats.items():
            print(f"   {key}: {value}")

    if errors:
        print(f"\n{Colors.ERROR}Errors found:{Colors.RESET}")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")

    if total_time is not None:
        print(f"\n{Colors.INFO}Total time: {total_time:.2f}s{Colors.RESET}")

    print()
