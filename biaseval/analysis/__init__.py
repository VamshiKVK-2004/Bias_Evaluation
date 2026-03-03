"""Analysis package."""

from biaseval.analysis.stereotype import run as run_stereotype


def run() -> None:
    """Entry function for analysis stage."""
    run_stereotype()
