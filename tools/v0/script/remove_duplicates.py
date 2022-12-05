'''
remove duplicate polynomial in a polynomial definition file
'''

import argparse
from pathlib import Path

from utility import hash_polynomial

# Command line input
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=Path, help="input polynomial definition file")
    parser.add_argument("-o","--output", type=Path, default=None, help="output polynomial definition file (default = `input`.out)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.input, 'r') as f: lines = f.readlines()
    if args.output is None:
        output = Path(args.input.stem + ".out")
    else:
        output = args.output
    with open(output, 'w') as f:
        seen_polynomials = {}
        for line in lines:
            hash_value = hash_polynomial(line)
            if hash_value not in seen_polynomials:
                seen_polynomials[hash_value] = 1
                print(line, end='', file=f)
