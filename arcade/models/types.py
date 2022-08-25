from pydantic import constr

UidType = constr(
    strip_whitespace=True,
    to_lower=True,
)