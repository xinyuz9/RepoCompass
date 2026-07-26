"""Authentication logic used by the repository-tool test fixture."""


def authenticate(username: str) -> bool:
    """Return whether the username identifies the fixture's allowed user."""

    return username == "fixture-user"
