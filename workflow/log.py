import logging
import sys


def setup_logging(
    logfile="pipeline.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return

    # ---- Console ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # ---- File ----
    fh = logging.FileHandler(logfile, mode="a")
    fh.setLevel(file_level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(ch)
    logger.addHandler(fh)
