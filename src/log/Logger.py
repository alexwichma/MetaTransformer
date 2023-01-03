"""
This module sets up logging for the training script.
"""


import logging


def init_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, 
                        level=logging.DEBUG, 
                        datefmt='%m/%d/%Y %H:%M:%S',
                        format='[%(asctime)s][%(levelname)s][%(name)s] - %(message)s')


def dict_to_str(d):
    entries = ["%s: %.4f" % (d_key.capitalize(), d[d_key]) for d_key in d]
    return ", ".join(entries)


def log_progress(logger: logging.Logger, curr_batch, metric_dict):
    logger.info("Batch %d - Metrics: %s", curr_batch, dict_to_str(metric_dict))


if __name__ == "__main__":
    init_logging("/share/ebuschon/data/test_log.txt")
    logger = logging.getLogger("myFancyModule")
    logger.info("Test info")
    logger.debug("Test debug")
    logger.error("Test error")