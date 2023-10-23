import json
import os
from math import isclose

import pytest


def prebuild_extenson_path():
    ext_path = os.getenv("CUSTOM_OP_LIB") or os.getenv("OV_TOKENIZER_PREBUILD_EXTENSION_PATH")
    if not ext_path:
        raise EnvironmentError(
            "No extension path found in the environment. "
            "Export path to libuser_ov_extensions.so to CUSTOM_OP_LIB or OV_TOKENIZER_PREBUILD_EXTENSION_PATH variable."
        )
    return ext_path


os.environ["OV_TOKENIZER_PREBUILD_EXTENSION_PATH"] = prebuild_extenson_path()
PASS_RATES_FILE = "pass_rates.json"


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus) -> None:
    """
    Tests fail if the test pass rate decreases
    """
    if exitstatus != pytest.ExitCode.TESTS_FAILED:
        return

    parent = os.path.commonprefix([item.nodeid for item in session.items]).strip("[]")

    with open(PASS_RATES_FILE) as f:
        previous_rates = json.load(f)

    pass_rate = 1 - session.testsfailed / session.testscollected
    previous = previous_rates.get(parent, 0)

    if isclose(pass_rate, previous):
        return

    if pass_rate > previous:
        session.exitstatus = pytest.ExitCode.OK
        previous_rates[parent] = pass_rate

        with open(PASS_RATES_FILE, "w") as f:
            json.dump(previous_rates, f, indent=4)
    else:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        reporter.write_line(f"Pass rate is lower! Current: {pass_rate}, previous: {previous}")
