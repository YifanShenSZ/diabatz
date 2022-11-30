'''
given a list of destination polynomials and a list of source polynomial parameters,
copy the source parameters to the destination polynomials
'''

import argparse
from pathlib import Path
from typing import Dict

def parse_args() -> argparse.Namespace: # command line input
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("polynomial", type=Path, help="polynomial file")
    parser.add_argument("parameter", type=Path, help="polynomial parameter file")
    parser.add_argument("-o","--output", type=Path, default=Path("parameter.out"), help="parameter output file (default = parameter.out)")
    args = parser.parse_args()
    return args

# assume less than 100 irreducibles and less than 100 coordinates per irreducible
def hash_polynomial(line: str) -> int:
    # edge case: bias
    if line.strip() == "bias": return 0
    # normal case
    strs = line.split()
    # remove comment
    i = 0
    while i < strs.__len__():
        if strs[i] == '#': break
        i += 1
    strs = strs[: i]
    # sort coordinates so that all permutations become the same
    strs.sort()
    # hash coordinates
    hash_value = 0
    weight = 100
    for irred_index in strs:
        irred, index = irred_index.split(',')
        hash_value += int(irred) * weight + int(index) * weight * 100
        weight *= 10000
    return hash_value

def read_polynomial(polynomial_file: Path) -> Dict:
    hashvalue2polynomial = {}
    with open(polynomial_file, 'r') as f: lines = f.readlines()
    for line in lines:
        if line != '\n':
            hash_value = hash_polynomial(line)
            hashvalue2polynomial[hash_value] = line
    return hashvalue2polynomial

def read_parameter(parameter_file: Path) -> Dict:
    hashvalue2parameter = {}
    with open(parameter_file, 'r') as f: lines = f.readlines()
    for i in range(0, len(lines), 2):
        hash_value = hash_polynomial(lines[i])
        parameter = float(lines[i + 1])
        if hash_value not in hashvalue2parameter:
            hashvalue2parameter[hash_value] = parameter
        else:
            hashvalue2parameter[hash_value] += parameter
    return hashvalue2parameter

if __name__ == "__main__":
    args = parse_args()
    hashvalue2polynomial = read_polynomial(args.polynomial)
    hashvalue2parameter = read_parameter(args.parameter)
    with open(args.output, 'w') as f:
        # print everything except bias
        for hashvalue in hashvalue2polynomial:
            if hashvalue != 0:
                polynomial = hashvalue2polynomial[hashvalue]
                if hashvalue in hashvalue2parameter:
                    parameter = hashvalue2parameter[hashvalue]
                else:
                    parameter = 0.0
                print(polynomial, end='', file=f)
                print("%25.15e" % parameter, file=f)
        # print bias if exists
        if 0 in hashvalue2parameter:
            print("bias", file=f)
            print("%25.15e" % hashvalue2parameter[0], file=f)
