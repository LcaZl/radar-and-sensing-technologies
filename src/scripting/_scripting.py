import argparse
import datetime
import logging
import pathlib
import sys
import typing
import yaml
import time
import psutil
import csv

def is_valid_file(path: str) -> str:
    """Identity function (i.e., the input is passed through and is not modified)
    that checks whether the given path is a valid file or not, raising an
    argparse.ArgumentTypeError if not valid.

    Parameters
    ----------
    path : str
        String representing a path to a file.

    Returns
    -------
    str
        The same string as in input

    Raises
    ------
    argparse.ArgumentTypeError
        An exception is raised if the given string does not represent a valid
        path to an existing file.
    """
    file = pathlib.Path(path)
    if not file.is_file:
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return path

def get_parser(description: str) -> argparse.ArgumentParser:
    """Function that generates the argument parser for the processor. Here, all
    the arguments and help message are defined.

    Parameters
    ----------
    description : str
        Text description to include in the help message of the script.

    Returns
    -------
    argparse.ArgumentParser
        The created argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--conf",
        "-c",
        dest="config_file_path",
        required=True,
        metavar="FILE",
        type=lambda x: is_valid_file(x),  # type: ignore
        help="The YAML configuration file.",
    )
    return parser

def logged_main(description: str, main_fn: typing.Callable) -> None:
    """Function that wraps around your main function adding logging capabilities
    and basic configuration with a yaml file passed with `-c <config_file_path>`

    Parameters
    ----------
    description : str
        Text description to include in the help message of the script.
    main_fn : typing.Callable
        Main function to execute
    """
    start_time = datetime.datetime.now()

    # ---- Parsing
    parser = get_parser(description)
    args = parser.parse_args()

    # ---- Loading configuration file
    config_file = args.config_file_path
    with open(config_file) as yaml_file:
        config = yaml.full_load(yaml_file)

    # ---- Config

    # Log folder
    log_folder = config["log_dir"]
    log_folder = pathlib.Path(log_folder)
    log_folder.mkdir(exist_ok=True, parents=True)
    # Log level
    log_level = config["log_level"]

    # ---- Logging

    # Change root logger level from WARNING (default) to NOTSET in order for all
    # messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)
    log_name = f"{main_fn.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}.log"
    log_filename = log_folder / log_name

    # Add stdout handler, with level defined by the config file (i.e., print log
    # messages to screen).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.getLevelName(log_level.upper()))
    frmttr_console = logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s")
    console_handler.setFormatter(frmttr_console)
    logging.getLogger().addHandler(console_handler)

    # Add file rotating handler, with level DEBUG (i.e., all DEBUG, WARNING and
    # INFO log messages are printed in the log file, regardless of the
    # configuration).
    logfile_handler = logging.FileHandler(filename=log_filename)
    logfile_handler.setLevel(logging.DEBUG)
    frmttr_logfile = logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s")
    logfile_handler.setFormatter(frmttr_logfile)
    logging.getLogger().addHandler(logfile_handler)

    logging.info("Configuration Complete.")

    # ---- Run tuning
    config["log_name"] = log_name
    main_fn(**config)

    # ---- Close Log

    logging.info(f"Close all. Execution time: {datetime.datetime.now()-start_time}")
    logging.getLogger().handlers.clear()
    logging.shutdown()
    
def monitor_memory(performance_file, interval=1):
    """
    Logs the memory usage every `interval` seconds and saves it to a CSV file.
    """
    process = psutil.Process()
    start_time = time.time()
    
    # Write the header for the CSV file
    with open(performance_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Elapsed Time (s)", "Memory Usage (MB)"])
    
    while True:
        elapsed_time = time.time() - start_time
        memory_info = process.memory_info()
        memory_used_mb = memory_info.rss / (1024 ** 2)  # Resident Set Size in MB

        # Append memory usage data to the CSV file
        with open(performance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([elapsed_time, memory_used_mb])
        
        time.sleep(interval)