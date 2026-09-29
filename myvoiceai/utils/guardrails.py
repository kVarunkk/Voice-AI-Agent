import re
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    category: str | None = None

# --- Input patterns ---
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"you are now",
    r"disregard your (system|previous) prompt",
    r"act as (a|an) (?!assistant)",
    r"reveal (your|the) (system prompt|instructions)",
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
}

BANNED_TOPICS_PATTERNS = [
    r"\b(kill|bomb|weapon|suicide|self[- ]harm)\b",
]

# --- Output patterns ---
OUTPUT_BLOCK_PATTERNS = [
    r"\b(fuck|shit|bitch)\b",
]

def _match_any(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return p
    return None

async def check_input(text: str) -> GuardrailResult:
    if m := _match_any(PROMPT_INJECTION_PATTERNS, text):
        return GuardrailResult(False, f"prompt_injection:{m}", "prompt_injection")
    if m := _match_any(BANNED_TOPICS_PATTERNS, text):
        return GuardrailResult(False, f"banned_topic:{m}", "banned_topic")
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            return GuardrailResult(False, f"pii:{label}", "pii")
    return GuardrailResult(True)

async def check_output(text: str) -> GuardrailResult:
    if m := _match_any(OUTPUT_BLOCK_PATTERNS, text):
        return GuardrailResult(False, f"toxic:{m}", "toxic")
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            return GuardrailResult(False, f"pii_leak:{label}", "pii")
    return GuardrailResult(True)