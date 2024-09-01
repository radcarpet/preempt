from babel.numbers import parse_decimal, format_currency, format_decimal
from decimal import Decimal
import re

def str2money(s: str, locale='en_US') -> Decimal:
    # Remove everything except numbers, commas, and a decimal point
    cleaned = re.sub(r'[^\d.,]', '', s)

    # Extract the numeric value using Babel's parse_decimal
    return parse_decimal(cleaned, locale)

def money2str(v: Decimal, orig: str, locale='en_US') -> str:
    # Extract currency symbols or other non-numeric characters
    prefix = re.search(r'^[^\d]+', orig)
    suffix = re.search(r'[^\d]+$', orig)

    prefix = prefix.group(0) if prefix else ""
    suffix = suffix.group(0) if suffix else ""

    # Strip any non-currency symbols (like commas) from prefix and suffix
    prefix = re.sub(r'[,\.\s]', '', prefix)
    suffix = re.sub(r'[,\.\s]', '', suffix)

    # Format the value using Babel's number formatting, then add the currency symbols back
    formatted_value = format_decimal(v, locale=locale)
    return f"{prefix}{formatted_value}{suffix}"
