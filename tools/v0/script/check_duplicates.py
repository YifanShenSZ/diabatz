'''
check if there is duplicate polynomial in a polynomial definition file
'''

import argparse
from pathlib import Path

from utility import hash_polynomial

# Command line input
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("file", type=Path, help="polynomial definition file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    hash_set = {}
    with open(args.file, 'r') as f: lines = f.readlines()
    for line in lines:
        hash_value = hash_polynomial(line)
        if hash_value in hash_set:
            print("duplicate line:")
            print(line)
        else:
            hash_set[hash_value] = 1
