import os
import argparse
from typing import List

def detect_failed_tests(logs_directory: str) -> List[str]:
    """Print any launched tests that did not finish""" 
    skip_line = "Skipping test:"
    start_line = "----- Running test: "
    finish_line = "----- Finished test: "

    # open log file and determine if file content should be deleted
    failed_logs = []
    for log in os.listdir(logs_directory):
        f = open(f"{logs_directory}/{log}", "r")
        log_text = f.read()
        f.close()
        
        # files are complete if they
        # 1) were skipped for having too large a d / n ratio, or
        # 2) have a starting and finishing line
        # otherwise, they are considered failed
        if not ((skip_line in log_text) or (start_line in log_text and finish_line in log_text)):
            failed_logs.append(log)

    failed_logs.sort()
    for log in failed_logs:
        print(log)

    return failed_logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    detect_failed_tests(args.dir)
