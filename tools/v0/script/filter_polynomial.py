'''
given a list of destination polynomials and a list of source polynomial parameters,
copy the source parameters to the destination polynomials
'''

import argparse
from pathlib import Path
from typing import Dict
import numpy as np

from utility import hash_polynomial

def parse_args() -> argparse.Namespace: # command line input
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("polynomial", type=Path, help="polynomial file")
    parser.add_argument("parameter", type=Path, help="polynomial parameter file")
    parser.add_argument("-o","--output", type=Path, default=Path("parameter.out"), help="parameter output file (default = parameter.out)")
    args = parser.parse_args()
    return args

def read_polynomial(polynomial_file: Path) -> Dict:
    hashvalue2polynomial = {}
    with open(polynomial_file, 'r') as f: lines = f.readlines()
    for line in lines:
        if line != '\n':
            hash_value = hash_polynomial(line)
            hashvalue2polynomial[hash_value] = line
    return hashvalue2polynomial

def read_parameter(parameter_file: Path) -> Dict:
    hashvalue2parameters = {}
    with open(parameter_file, 'r') as f: lines = f.readlines()
    for i in range(0, len(lines), 2):
        hash_value = hash_polynomial(lines[i])
        parameter_strings = lines[i + 1].split()
        parameters = []
        for parameter_string in parameter_strings:
            parameters.append(float(parameter_string))
        parameters = np.array(parameters)
        if hash_value not in hashvalue2parameters:
            hashvalue2parameters[hash_value] = parameters
        else:
            hashvalue2parameters[hash_value] += parameters
    return hashvalue2parameters

if __name__ == "__main__":
    args = parse_args()
    hashvalue2polynomial = read_polynomial(args.polynomial)
    hashvalue2parameters = read_parameter(args.parameter)
    nhiddens = len(hashvalue2parameters[[*hashvalue2parameters.keys()][0]])
    with open(args.output, 'w') as f:
        # print everything except bias
        for hashvalue in hashvalue2polynomial:
            if hashvalue != 0:
                polynomial = hashvalue2polynomial[hashvalue]
                if hashvalue in hashvalue2parameters:
                    parameters = hashvalue2parameters[hashvalue]
                else:
                    parameters = np.zeros(nhiddens, dtype=float)
                print(polynomial, end='', file=f)
                for i in range(len(parameters)):
                    print("%25.15e" % parameters[i], end='', file=f)
                print(file=f)
        # print bias if exists
        if 0 in hashvalue2parameters:
            print("bias", file=f)
            parameters = hashvalue2parameters[0]
            for i in range(len(parameters)):
                print("%25.15e" % parameters[i], end='', file=f)
            print(file=f)
