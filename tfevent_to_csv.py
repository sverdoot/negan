#!/usr/bin/env python3

"""
This script exctracts training variables from all logs from
tensorflow event files ("event*"), writes them to Pandas
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.
The magic "5" infers there are only the following v.tags:
[lr, loss, acc, val_loss, val_acc]
"""

import argparse
from pathlib import Path

import pandas as pd
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    args = parser.parse_args()
    return args


def main(args):
    event_paths = list(map(lambda x: x.as_posix(), Path(args.log_dir).glob("event*")))

    # Extraction function
    def sum_log(path):
        runlog = pd.DataFrame(columns=["metric", "value"])
        try:
            for e in tf.train.summary_iterator(path):
                for v in e.summary.value:
                    r = {"metric": [v.tag], "value": [v.simple_value]}
                    runlog = runlog.append(r, ignore_index=True)
                    # pd.concat([runlog, pd.DataFrame.from_records([r])])

        # Dirty catch of DataLossError
        except RuntimeError:
            print(f"Event file possibly corrupt: {path}")
            return None

        return runlog

    # Call & append
    all_log = pd.DataFrame()
    for path in event_paths:
        log = sum_log(path)
        if log is not None:
            if all_log.shape[0] == 0:
                all_log = log
            else:
                all_log = all_log.append(log)

    # Inspect
    print(all_log.shape)
    all_log.head()

    # Store
    all_log.to_csv(Path(args.log_dir, "logs.csv"), index=None)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
