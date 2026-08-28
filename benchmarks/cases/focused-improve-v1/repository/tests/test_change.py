from pkg.example import allowed


def test_allowed() -> None:
    assert allowed() == 2
