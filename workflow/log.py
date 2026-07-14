import logging
import sys
import os


def setup_logging(
    logfile="pipeline.log",
    tuning_logfile=None,
    console_level=logging.INFO,
    file_level=logging.DEBUG,
):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Console ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)

    # ---------- Main log ----------
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # ---------- Optional tuning log ----------
    if tuning_logfile:

        os.makedirs(os.path.dirname(tuning_logfile), exist_ok=True)

        tuning_handler = logging.FileHandler(
            tuning_logfile,
            mode="w",
        )

        tuning_handler.setLevel(file_level)
        tuning_handler.setFormatter(formatter)

        tuning_handler.addFilter(lambda record: record.name == "w2v_tuning")

        logger.addHandler(tuning_handler)
