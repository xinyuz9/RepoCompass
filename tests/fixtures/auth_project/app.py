"""Application code used by the repository-tool test fixture."""

from auth import authenticate


def login(username: str) -> bool:
    """Delegate a login attempt to the authentication implementation."""

    return authenticate(username)
