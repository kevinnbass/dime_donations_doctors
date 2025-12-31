"""
Text normalization utilities for name and address standardization.

Used for record linkage between DIME and NPPES datasets.
"""

import re
import unicodedata
from functools import lru_cache
from typing import Optional

# Common name suffixes to strip
NAME_SUFFIXES = {
    "jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v",
    "md", "m.d.", "m.d", "do", "d.o.", "d.o",
    "phd", "ph.d.", "ph.d", "mph", "m.p.h.",
    "mba", "m.b.a.", "jd", "j.d.",
    "rn", "r.n.", "np", "n.p.", "pa", "p.a.",
    "esq", "esq.",
}

# Common name prefixes
NAME_PREFIXES = {"dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss"}

# State abbreviation mapping
STATE_ABBREVS = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Reverse lookup
STATE_NAMES = {v: k for k, v in STATE_ABBREVS.items()}

# Street type abbreviations for address normalization
STREET_ABBREVS = {
    "avenue": "ave", "boulevard": "blvd", "circle": "cir", "court": "ct",
    "drive": "dr", "expressway": "expy", "freeway": "fwy", "highway": "hwy",
    "lane": "ln", "parkway": "pkwy", "place": "pl", "road": "rd",
    "square": "sq", "street": "st", "terrace": "ter", "trail": "trl",
    "way": "way", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "apt", "building": "bldg", "floor": "fl", "suite": "ste",
    "unit": "unit", "room": "rm", "department": "dept",
}


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to ASCII equivalents.

    Args:
        text: Input text

    Returns:
        ASCII-normalized text
    """
    if not text:
        return ""
    # Normalize to NFD form and remove diacritics
    normalized = unicodedata.normalize("NFD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace characters to single spaces and strip.

    Args:
        text: Input text

    Returns:
        Whitespace-normalized text
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def remove_punctuation(text: str, keep_chars: str = "") -> str:
    """
    Remove punctuation from text, optionally keeping specified characters.

    Args:
        text: Input text
        keep_chars: Characters to keep (e.g., "-" for hyphenated names)

    Returns:
        Text with punctuation removed
    """
    if not text:
        return ""
    pattern = rf"[^\w\s{re.escape(keep_chars)}]"
    return re.sub(pattern, " ", text)


@lru_cache(maxsize=10000)
def normalize_name(name: str, remove_suffixes: bool = True) -> str:
    """
    Normalize a person's name for matching.

    Steps:
    1. Convert to lowercase
    2. Normalize unicode to ASCII
    3. Remove punctuation (keep hyphens for hyphenated names)
    4. Remove common prefixes (Dr., Mr., etc.)
    5. Remove common suffixes (Jr., MD, PhD, etc.)
    6. Collapse whitespace

    Args:
        name: Input name string
        remove_suffixes: Whether to remove credential/generational suffixes

    Returns:
        Normalized name string
    """
    if not name or not isinstance(name, str):
        return ""

    # Lowercase and normalize unicode
    result = normalize_unicode(name.lower())

    # Remove punctuation but keep hyphens
    result = remove_punctuation(result, keep_chars="-")

    # Tokenize
    tokens = result.split()

    # Remove prefixes
    while tokens and tokens[0].rstrip(".") in NAME_PREFIXES:
        tokens = tokens[1:]

    # Remove suffixes
    if remove_suffixes:
        while tokens and tokens[-1].rstrip(".") in NAME_SUFFIXES:
            tokens = tokens[:-1]

    # Rejoin and normalize whitespace
    return normalize_whitespace(" ".join(tokens))


def extract_name_parts(full_name: str) -> dict[str, str]:
    """
    Extract first name, middle name/initial, and last name from a full name.

    Handles formats like:
    - "JOHN SMITH"
    - "SMITH, JOHN"
    - "JOHN A SMITH"
    - "JOHN A. SMITH, MD"

    Args:
        full_name: Full name string

    Returns:
        Dictionary with keys: first, middle, last (all normalized)
    """
    if not full_name:
        return {"first": "", "middle": "", "last": ""}

    # Normalize first
    name = normalize_name(full_name)

    # Check for "Last, First" format
    if "," in full_name.lower():
        parts = name.replace(",", "").split()
        if len(parts) >= 2:
            last = parts[0]
            first = parts[1]
            middle = " ".join(parts[2:]) if len(parts) > 2 else ""
            return {"first": first, "middle": middle, "last": last}

    # Standard "First [Middle] Last" format
    parts = name.split()
    if len(parts) == 0:
        return {"first": "", "middle": "", "last": ""}
    elif len(parts) == 1:
        return {"first": "", "middle": "", "last": parts[0]}
    elif len(parts) == 2:
        return {"first": parts[0], "middle": "", "last": parts[1]}
    else:
        return {"first": parts[0], "middle": " ".join(parts[1:-1]), "last": parts[-1]}


@lru_cache(maxsize=10000)
def normalize_address(address: str) -> str:
    """
    Normalize a street address for matching.

    Steps:
    1. Convert to lowercase
    2. Normalize unicode
    3. Remove punctuation
    4. Standardize street type abbreviations
    5. Standardize directionals

    Args:
        address: Input address string

    Returns:
        Normalized address string
    """
    if not address or not isinstance(address, str):
        return ""

    result = normalize_unicode(address.lower())
    result = remove_punctuation(result)
    result = normalize_whitespace(result)

    # Standardize abbreviations
    tokens = result.split()
    normalized_tokens = []
    for token in tokens:
        normalized = STREET_ABBREVS.get(token, token)
        normalized_tokens.append(normalized)

    return " ".join(normalized_tokens)


@lru_cache(maxsize=5000)
def normalize_city(city: str) -> str:
    """
    Normalize a city name for matching.

    Args:
        city: City name

    Returns:
        Normalized city name
    """
    if not city or not isinstance(city, str):
        return ""

    result = normalize_unicode(city.lower())
    result = remove_punctuation(result)
    result = normalize_whitespace(result)

    # Common abbreviations
    result = re.sub(r"\bst\b", "saint", result)
    result = re.sub(r"\bft\b", "fort", result)
    result = re.sub(r"\bmt\b", "mount", result)

    return result


def normalize_state(state: str) -> str:
    """
    Normalize a state to its 2-letter abbreviation.

    Args:
        state: State name or abbreviation

    Returns:
        2-letter state abbreviation, or empty string if not recognized
    """
    if not state or not isinstance(state, str):
        return ""

    state_clean = state.strip().upper()

    # Already an abbreviation
    if len(state_clean) == 2 and state_clean in STATE_NAMES:
        return state_clean

    # Full name
    state_lower = state.strip().lower()
    return STATE_ABBREVS.get(state_lower, "")


def normalize_zip(zipcode: str) -> str:
    """
    Extract 5-digit ZIP code from various formats.

    Handles:
    - "12345"
    - "12345-6789"
    - "123456789"

    Args:
        zipcode: ZIP code string

    Returns:
        5-digit ZIP code, or empty string if invalid
    """
    if not zipcode:
        return ""

    # Extract digits only
    digits = re.sub(r"\D", "", str(zipcode))

    # Return first 5 digits if valid
    if len(digits) >= 5:
        return digits[:5]

    return ""


def get_zip_prefix(zipcode: str, n_digits: int = 3) -> str:
    """
    Get the first N digits of a ZIP code for blocking.

    Args:
        zipcode: ZIP code string
        n_digits: Number of prefix digits (default 3)

    Returns:
        ZIP prefix
    """
    normalized = normalize_zip(zipcode)
    return normalized[:n_digits] if len(normalized) >= n_digits else ""


@lru_cache(maxsize=10000)
def soundex(name: str) -> str:
    """
    Compute Soundex code for a name.

    Used for phonetic blocking in record linkage.

    Args:
        name: Input name

    Returns:
        4-character Soundex code
    """
    if not name:
        return ""

    # Normalize input
    name = normalize_name(name).upper()
    if not name:
        return ""

    # Soundex mapping
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    # Keep first letter
    first_letter = name[0] if name else ""
    if not first_letter.isalpha():
        return ""

    # Convert remaining letters
    codes = [first_letter]
    prev_code = mapping.get(first_letter, "")

    for char in name[1:]:
        code = mapping.get(char, "")
        if code and code != prev_code:
            codes.append(code)
        prev_code = code if code else prev_code

    # Pad or truncate to 4 characters
    result = "".join(codes)
    result = result[:4].ljust(4, "0")

    return result


def extract_credentials_from_name(name: str) -> list[str]:
    """
    Extract professional credentials from a name string.

    Args:
        name: Full name potentially including credentials

    Returns:
        List of credential strings found
    """
    if not name:
        return []

    credentials = []
    name_upper = name.upper()

    # Common credential patterns (case-insensitive)
    patterns = [
        r"\bM\.?D\.?\b",
        r"\bD\.?O\.?\b",
        r"\bPH\.?D\.?\b",
        r"\bJ\.?D\.?\b",
        r"\bM\.?B\.?A\.?\b",
        r"\bR\.?N\.?\b",
        r"\bN\.?P\.?\b",
        r"\bP\.?A\.?\b",
        r"\bD\.?D\.?S\.?\b",
        r"\bD\.?M\.?D\.?\b",
        r"\bD\.?V\.?M\.?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, name_upper, re.IGNORECASE)
        if match:
            credentials.append(match.group().replace(".", "").upper())

    return credentials
