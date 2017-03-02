#!/usr/bin/env python

from __future__ import print_function

import sys
import logging
import ruamel.yaml
import signal
import cnn_image_processing as ci


signal.signal(signal.SIGINT, lambda x, y: sys.exit(1))
LOGGER = logging.getLogger("cnn_image_processing")


def parse_args():
    print(' '.join(sys.argv))
    import argparse
    parser = argparse.ArgumentParser(description="Train the cnn")

    parser.add_argument("-c", "--config-file", required=True,
                        help="YAML configuration file.")
    parser.add_argument("-v", "--verbose", action="store_true", required=False,
                        help="Set the verbose mode.")

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # Initialize logging
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)

    logging.basicConfig()

    # Open, parse and print the configuration file
    with open(args.config_file) as file:
        config = ruamel.yaml.load(file, ruamel.yaml.RoundTripLoader)
        LOGGER.info(ruamel.yaml.dump(
            config, Dumper=ruamel.yaml.RoundTripDumper))

    LOGGER.info(" Parsing config.")
    app = ci.Creator.parse_config(config)

    LOGGER.info(" Starting processes.")
    for process in app[1:]:
        process.start()

    app[0].run()


if __name__ == "__main__":
    main()
