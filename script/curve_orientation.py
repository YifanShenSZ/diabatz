'''
check if the even-order pure terms have a positive coefficient
'''

import argparse
from pathlib import Path
from typing import List, Tuple

# Command line input
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", type=Path, help="interpretable parameter file")
    args = parser.parse_args()
    return args

def line2coords(line:str) -> List[Tuple[int]]:
    strs = line.split()
    # remove comment
    i = 0
    while i < strs.__len__():
        if strs[i] == '#': break
        i += 1
    strs = strs[: i]
    # string -> coordinate tuple
    coords = []
    for str in strs:
        irred, index = str.split(',')
        coords.append((int(irred), int(index)))
    return coords

if __name__ == "__main__":
    args = parse_args()
    hash_set = {}
    with open(args.input, 'r') as f: lines = f.readlines()
    for i in range(0, len(lines), 2):
        if lines[i].strip() == "bias": break
        coords = line2coords(lines[i])
        # count unique coordinates and their repeats
        coord_repeats = {}
        for coord in coords:
            if coord not in coord_repeats:
                coord_repeats[coord] = 1
            else:
                coord_repeats[coord] += 1
        # check if all unique coordinates have even repeats
        orientation_determiner = True
        for value in coord_repeats.values():
            if value % 2 != 0:
                orientation_determiner = False
                break
        # output if negative
        if orientation_determiner:
            coeff = float(lines[i + 1])
            if coeff < 0:
                print(lines[i], coeff)

