import json
import os
from math import isclose
from pathlib import Path
from io import StringIO

import pytest

PASS_RATES_FILE = Path(__file__).parent / "pass_rates.json"


def build_coverege_report(session: pytest.Session) -> None:
    import pandas as pd
    import numpy as np
    from pytest_harvest import get_session_results_df

    def add_tokenizer_type(row):
        if not pd.isnull(row["wordpiece_tokenizers_param"]):
            return "wordpiece"
        if not pd.isnull(row["bpe_tokenizers_param"]):
            return "bpe"
        if not pd.isnull(row["sentencepice_tokenizers_param"]):
            return "sentencepiece"
        if not pd.isnull(row["tiktoken_tokenizers_param"]):
            return "tiktoken"

    results_df = get_session_results_df(session)
    results_df["tokenizer_type"] = results_df.apply(add_tokenizer_type, axis=1)
    results_df.wordpiece_tokenizers_param.fillna(results_df.bpe_tokenizers_param, inplace=True)
    results_df.wordpiece_tokenizers_param.fillna(results_df.sentencepice_tokenizers_param, inplace=True)
    results_df.wordpiece_tokenizers_param.fillna(results_df.tiktoken_tokenizers_param, inplace=True)
    results_df.is_fast_tokenizer_param.fillna(True, inplace=True)
    results_df.status = (results_df.status == "passed").astype(int)
    results_df["models"] = results_df.wordpiece_tokenizers_param + results_df.is_fast_tokenizer_param.apply(lambda x: "" if x else "_slow")

    final = results_df[["tokenizer_type", "models", "test_string", "status"]]
    grouped_by_model = final.groupby(["tokenizer_type", "models"]).agg("mean")
    grouped_by_type = final.groupby(["tokenizer_type"]).agg("mean")

    styled = grouped_by_type.style.background_gradient(cmap="RdBu", axis=0)

    readme_path = Path("../README.md")
    with open(readme_path) as f:
        old_readme = f.read().split("## Test Coverage")[0]

    new_readme = StringIO()
    new_readme.write(old_readme)
    new_readme.write("## Test Coverage\n\n")
    new_readme.write("### Covarage by Tokenizer Type\n")
    styled.to_html(new_readme)
    new_readme.write("\n ### Covarage by Model Type\n")

    styled = grouped_by_model.style.background_gradient(cmap="RdBu", axis=0)
    styled.to_html(new_readme)

    with open(readme_path, "w") as f:
        f.write(new_readme.getvalue())


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: pytest.ExitCode) -> None:
    """
    Tests fail if the test pass rate decreases
    """
    build_coverege_report(session)
    if exitstatus != pytest.ExitCode.TESTS_FAILED:
        return

    parent = os.path.commonprefix([item.nodeid for item in session.items]).strip("[]")

    with open(PASS_RATES_FILE) as f:
        previous_rates = json.load(f)

    pass_rate = 1 - session.testsfailed / session.testscollected
    previous = previous_rates.get(parent, 0)

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if isclose(pass_rate, previous):
        session.exitstatus = pytest.ExitCode.OK
        reporter.write_line(f"New pass rate isclose to previous: {pass_rate}")
        return

    if pass_rate > previous:
        reporter.write_line(f"New pass rate {pass_rate} is bigger then previous: {previous}")
        session.exitstatus = pytest.ExitCode.OK
        previous_rates[parent] = pass_rate

        with open(PASS_RATES_FILE, "w") as f:
            json.dump(previous_rates, f, indent=4)
    else:
        reporter.write_line(f"Pass rate is lower! Current: {pass_rate}, previous: {previous}")
