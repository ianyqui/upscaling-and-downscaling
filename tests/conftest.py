import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="roda também os testes lentos (DA completo, 75 codebooks por problema)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: teste lento")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="lento; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
