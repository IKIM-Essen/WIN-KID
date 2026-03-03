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

    if logger.handlers:
        return

    # ---- Console ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # ---- Main File ----
    log_dir = os.path.dirname(logfile)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(logfile, mode="a")
    fh.setLevel(file_level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(ch)
    logger.addHandler(fh)

    # ---- Optional Tuning File ----
    if tuning_logfile is not None:
        os.makedirs(os.path.dirname(tuning_logfile), exist_ok=True)

        tuning_handler = logging.FileHandler(tuning_logfile, mode="a")
        tuning_handler.setLevel(file_level)
        tuning_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Only accept records from w2v_tuning logger
        tuning_handler.addFilter(lambda record: record.name == "w2v_tuning")

        logger.addHandler(tuning_handler)
