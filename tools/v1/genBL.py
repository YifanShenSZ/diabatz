'''
given number of atoms, generate all possible bond lengthes
output to an internal coordinate definition file
'''

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace: # Command line input
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("NAtoms", type=int, help="number of atoms")
    parser.add_argument("-o","--output", type=Path, default=Path("bond-lengthes.IntCoordDef"), help="output internal coordinate definition file (default = bond-lengthes.IntCoordDef)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.output, 'w') as f:
        count = 1
        for i in range(args.NAtoms):
            for j in range(i + 1, args.NAtoms):
                print("%6d    1.000000    stretching%6d%6d" % (count, i + 1, j + 1), file=f)
                count += 1
