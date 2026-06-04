from datetime import datetime
from loguru import logger
import sys


def setup_logger(log_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> - {message}",
        colorize=True,
    )
    logger.add(
        "logs/robot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        encoding="utf-8",
    )


def pct_change(current: float, reference: float) -> float:
    if reference == 0:
        return 0
    return round((current - reference) / reference * 100, 2)


def format_currency(value: float, currency: str = "TL") -> str:
    return f"{value:,.2f} {currency}"


def now_str() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")
