from __future__ import annotations

import os
import pathlib
import subprocess

import click

from jijbench.consts.default import DEFAULT_RESULT_DIR


@click.command()
@click.option(
    "--logdir", default=DEFAULT_RESULT_DIR, help="Directory to save benchmark results"
)
def main(logdir: str):
    os.environ["logdir"] = logdir
    os.environ["STREAMLIT_THEME_BACKGROUND_COLOR"] = "white"

    command = ["streamlit", "run", f"{pathlib.Path(__file__).parent}/app.py"]
    subprocess.run(command)


if __name__ == "__main__":
    main()
