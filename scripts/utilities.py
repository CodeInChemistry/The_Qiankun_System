#!/usr/bin/env python3

"""
Script Name: utilities.py
Version: 1.0.1
Author: Shihong CHEN<shihong.chen@connect.ust.hk>
Date: 2025-06-23
"""

import re
import os
import sys
import time
import json
import copy
import random
import reedsolo
import argparse
import numpy as np
import pandas as pd
import configparser
from Bio import Align
import Levenshtein as Lev
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
# plt.style.use('ggplot')

class WELL_N_RNG:  
    """
    A Random number generator with a N-bit range based on WELL algorithm.
    The N can specify by the num_bits.
    """
    
    def __init__(self, seed=0, prime=1812433257, state_size=16, num_bits=64):
        """Initialize the WELLN generator with a seed and an optional custom multiplier."""
        self.w = state_size  # State size
        self.n = num_bits  # Number of bits
        self.index = 0  # Current index
        self.state = [0] * 2  # State array
        self.range_top = 2**16-1
        self.multiplier = prime
        if seed in [None, 0]:
            self._seed = int(time.time())
        else:
            self._seed = seed
        self.initialize_generator()
        
        
    def initialize_generator(self):
        """Initialize the generator with a seed and multiplier."""
        
        self.index = self.w - 1  # Current index
        self.state = [0] * self.w  # State array
        self.range_top = 2**self.n-1
        
        self.state[0] = self._seed & self.range_top
        for i in range(1, self.w):
            self.state[i] = (self.multiplier * (self.state[i - 1] ^ (self.state[i - 1] >> 30)) + i) & self.range_top

    def seed(self, seed=None):
        self.update(seed=seed)
    
    def update(self, **kwargs):
        
        for kw, value in kwargs.items():
            if kw in ['w', 'state_size']:
                self.w = value
            elif kw in ['n', 'num_bits']:
                self.n = value
            elif kw in ['m', 'prime']:
                self.multiplier = value
            elif kw in ['s', 'seed']:
                self._seed = value
            else:
                sys.stderr.write(f'{kw} is not accetible, ignore this param\n')
                continue
        self.initialize_generator()
    
    def _next(self):
        """Generate the next random number using the WELL algorithm."""
        self.index = (self.index + 1) % self.w
        z0 = self.state[self.index]
        z1 = self.state[(self.index + 1) % self.w]
        z2 = self.state[(self.index + 2) % self.w]
        z3 = self.state[(self.index + 3) % self.w]

        # Update the state using the WELL algorithm
        self.state[self.index] = (
            z0 ^ ((z0 << 5) & self.range_top) ^ ((z1 << 1) & self.range_top) 
            ^ z2 ^ ((z2 >> 2) & self.range_top) ^ z3
        )

        return self.state[self.index] & self.range_top
    
    def get_binary(self, size=1):
        return [self._next() % 2 for _ in range(size)]
    
    def choice(self, size, options):
        option_size = len(options)
        return [options[self._next() % option_size] for _ in range(size)]
        
    def random_integer(self, low, high):
        """Generate a random integer between low and high (inclusive)."""
        return low + self._next() % (high - low + 1)

    def random_float(self, low, high):
        """Generate a random float between low and high (inclusive)."""
        return low + (self._next() / self.range_top) * (high - low)

# definition of YYC codec
class YYC:
    """
    The Yin-Yang codec system, which transform binary information to DNA 
    sequence or reverse.
    """
    def __repr__(self) -> str:
        rz_str = f'Yin-Yang codec {self.code_sys_idx}'.center(30, '=')+'\n'
        rz_str += f'Yang codec'.center(30, '-')+'\n'
        rz_str += f'{[b for b in self.negative]} <=> 0'.center(30, ' ')+'\n'
        rz_str += f'{[b for b in self.positive]} <=> 1'.center(30, ' ')+'\n'
        rz_str += f'Yin codec'.center(30, '-')+'\n'
        rz_str += 'prebase'.center(11, '.') + 'output-base'.center(19, '~') + '\n'
        rz_str += '%6s%6s%6s%6s%6s\n' % ('', self.negative[0], self.negative[1], self.positive[0], self.positive[1])
        rz_str += '%6s%6d%6d%6d%6d\n' % ('A', abs(int(self.positive_idx[0]) -1), int(self.positive_idx[0]), abs(int(self.positive_idx[1]) -1), int(self.positive_idx[1]))
        rz_str += '%6s%6d%6d%6d%6d\n' % ('T', abs(int(self.positive_idx[2]) -1), int(self.positive_idx[2]), abs(int(self.positive_idx[3]) -1), int(self.positive_idx[3]))
        rz_str += '%6s%6d%6d%6d%6d\n' % ('C', abs(int(self.positive_idx[4]) -1), int(self.positive_idx[4]), abs(int(self.positive_idx[5]) -1), int(self.positive_idx[5]))
        rz_str += '%6s%6d%6d%6d%6d\n\n' % ('G', abs(int(self.positive_idx[6]) -1), int(self.positive_idx[6]), abs(int(self.positive_idx[7]) -1), int(self.positive_idx[7]))
        return rz_str

    def __init__(self, code_sys_idx=876):
        ne_base = ['AT', 'AC', 'AG', 'CT', 'CG', 'TG']
        po_base = ['GC', 'TG', 'TC', 'GA', 'TA', 'AC']
        self.code_sys_idx = code_sys_idx % 1536

        set_idx = int(self.code_sys_idx / 256) # select the Yang codec scheme
        conbination_idx = self.code_sys_idx % 256 # select the Yin codec scheme
        
        bidx = bin(conbination_idx)[2:]
        positive_idx = '0' * (8-len(bidx)) + bidx

        self.negative = ne_base[set_idx]
        self.positive = po_base[set_idx]

        self.positive_idx = positive_idx
    
    def __call__(self, sa, sb, pb='A'):
        return self.encode(sa, sb, pb)
    
    def encode(self, sa, sb, pseudobase='A'):
        pre = pseudobase
        dna = ''
        for i in range(len(sa)):
            a = sa[i]
            b = sb[i]
            if pre == 'A':
                r_idex = 0
            elif pre == 'T':
                r_idex = 1
            elif pre == 'C':
                r_idex = 2
            elif pre == 'G':
                r_idex = 3
            else:
                raise Exception(f'UnExpected pre base "{pre}"\n')
            
            if a in [0, '0']:
                r_idex *= 2
                selector = self.negative
            else:
                r_idex = 2 * r_idex + 1
                selector = self.positive

            if b in [0, '0']:
                idx = abs(int(self.positive_idx[r_idex]) -1)
            else:
                idx = int(self.positive_idx[r_idex])
            
            dna += selector[idx]
            pre = selector[idx]
        return dna
    
    def decode(self, dna, pseudobase='A'):
        pre = pseudobase
        has = ''
        hbs = ''
        for base in dna:
            if pre == 'A':
                r_idex = 0
            elif pre == 'T':
                r_idex = 1
            elif pre == 'C':
                r_idex = 2
            elif pre == 'G':
                r_idex = 3
            else:
                raise Exception(f'Unexpected pre base "{pre}"\n')
            
            if base in self.negative:
                has += '0'
                r_idex = 2 * r_idex
                selector = self.negative
            else:
                r_idex = 2 * r_idex + 1
                has += '1'
                selector = self.positive
            
            if selector[int(self.positive_idx[r_idex])] == base:
                hbs += '1'
            else:
                hbs += '0'
            pre = base

        return has, hbs
    
def XOR(its):
    return its.count(1) % 2

def XOR2(bi_infos):
    """
    calculate XOR to 2D matrix in vectically direction
    """
    rz = np.sum([[int(b) for b in bi] for bi in bi_infos], axis=0) % 2
    
    return ''.join([str(b) for b in rz])

class HammingCode:
    """
    encode or decode binary information via Hamming code algorithm
    """
    def __init__(self, hc_type:str='8-4') -> None:
        """
        Argument:
        hc_type -- hamming code type, default is 8,4 hamming code
                   valid types are '8-4', '16-11', '32-26', '64-57'
                   This hamming code can detect less than 2 errors and correct 1 error

                   {total length}-{information size} or 
                   {tgt size}-{src size}
        """
        
        self.hc_type = hc_type
        items = hc_type.split('-')
        self.src_size = int(items[1])
        self.tgt_size = int(items[0])
        self.rank = int(np.log2(self.tgt_size))
        self.parities = [int(np.power(2, idx)-1) for idx in range(self.rank+1)]
    
    def __repr__(self) -> str:
        return f'HammingCode ({self.hc_type})\nsrc_size:{self.src_size}\ntgt_size:{self.tgt_size}\ncoding_distribution:{"".join(["x"+"."*(2**i-1) for i in range(self.rank)])}x'

    def encode(self, source_bits):
        # check source bits length
        if len(source_bits) != self.src_size:
            raise Exception(f'Error: Input source bits is invalid shape for "{self.hc_type}" hamming code\n')
        
        tgt = [np.nan]
        src = [int(bit) for bit in source_bits]
        idx = 0

        for i in range(1, self.rank):
            tgt.append(np.nan)
            tgt += src[idx:idx+2**i-1]
            idx += 2**i-1

        for i in range(self.rank):
            items = []
            for j in range(2**(i)-1, len(tgt), 2 ** (i+1)):
                items += tgt[j:j+2**i]
            tgt[2**i-1] = XOR(items)

        tgt.append(XOR(tgt))
        return ''.join([str(bit) for bit in tgt])
    
    def decode(self, encoded_bits):
        
        # check input bits size
        if len(encoded_bits) != self.tgt_size:
            raise Exception(f'Error: Input encoded bits is invalid shape for "{self.hc_type}" hamming code\n')

        tgt = [int(b) for b in encoded_bits]
        parities_check = []
        for i in range(self.rank):
            items = []
            for j in range(2**(i)-1, len(tgt), 2 ** (i+1)):
                items += tgt[j:j+2**i]      
            parities_check.append(XOR(items))
        
        error_position = int(np.sum([parities_check[i] * 2**i for i in range(len(parities_check))]))
        even_check = XOR(tgt)
        
        if error_position != 0:
            if even_check == 0:
                return '', -1
                # raise Exception('There are two error positions in the encoded bits, can not correct the code\n')
            else:
                tgt[error_position-1] = 1- tgt[error_position-1]
        else:
            if even_check == 1:
                # error occured at the even parity bit
                tgt[-1] = 1- tgt[-1]
                error_position = len(tgt)
        
        decoded_info = [tgt[idx] for idx in range(self.tgt_size) if idx not in self.parities]
        return ''.join([str(bit) for bit in decoded_info]), error_position

# factorial YYC
def factorial(n):
    """
    get factorial number of n
    """
    # calculate factorial of n
    precount_values = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200,
                       1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000,
                       51090942171709440000, 1124000727777607680000, 25852016738884976640000, 620448401733239439360000,
                       15511210043330985984000000, 403291461126605635584000000, 10888869450418352160768000000,
                       304888344611713860501504000000, 8841761993739701954543616000000
                      ]
    if n < 0:
        raise Exception(f'n must be an integer that not less than 0\n')
    elif 0 <= n < 30:
        return precount_values[n]
    else:
        return n * factorial(n - 1)
    
def decimal_to_factorial_num(dc_idx, base=10):
    if base is None:
        for i in range(30):
            if dc_idx / factorial(i) >= i:
                continue
            else:
                base = i + 1
                break
    elif dc_idx >= factorial(base):
        raise Exception(f'The decimal index ({dc_idx}) is too large for a specific base {base}, because it is greater than facotrial({base})\n')
    
    if base is None:
        raise Exception(f'The decimal index ({dc_idx}) is too large, greater than facotrial(30)\n')
    
    rest = dc_idx
    fi = []
    for i in range(base-1, 0, -1):
        fi.append(rest // factorial(i))
        rest = rest % factorial(i)
    fi.append(0)
    return fi

def factorial_num_to_decimal(fidx):
    # transform factorial index to decimal index 
    return int(np.sum([factorial(i) * fidx[-1-i] for i in range(len(fidx))], dtype=int))

def order_seq_to_ftnum(order_seq):
    # transform the order of reverse and complement fragments into factorial index
    ftnum = []
    tmpseq = copy.deepcopy(order_seq)
    for i in range(len(tmpseq)):
        idx = tmpseq.index(i)
        ftnum.append(idx)
        tmpseq.pop(idx)
    return ftnum

def ftnum_to_order_seq(ftnum):
    # transform factorial index into the order of reverse and complement fragments
    order_seq = []
    for i in range(len(ftnum)-1, -1, -1):
        order_seq.insert(ftnum[i], i)
    return order_seq

# definition of IO functions
def get_binary_info(file_path):
    """
    Read a file and get binary information

    not support for very large file now.
    """
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    
    binary_info = ''.join(format(byte, '08b') for byte in binary_data)
    return binary_info

def save_binary_info(binary_seq, byte_num, output_file):
    """
    transform a binary sequence to bytes and write into a file
    """
    def binary_str_to_byte(binary_str):
        byte_value = 0

        for i in range(8):
            byte_value += 2**i * int(binary_str[-i-1])

        return byte_value

    output_str = bytearray()
    if byte_num is None:
        byte_num = len(binary_seq) // 8
    for idx in range(0, byte_num):
        output_str.append(binary_str_to_byte(binary_seq[idx*8:idx*8+8]))
        # print(binary_seq[idx*8:idx*8+8])
        # output_str.append(int(binary_seq[idx*8:idx*8+8], 2))

    with open(output_file, 'wb') as file_handle:
        file_handle.write(output_str[:byte_num])
        file_handle.close()

def reverse_complemnet(dna_seq):
    # generate reverse and complement DNA fragment
    T = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G',
        'N': 'N',
        '-': '-'
    }
    
    return ''.join([T[b] for b in dna_seq.upper()][::-1])

# functions for simulations
def get_random_dna_seq(size):
    return ''.join([random.choice(['A', 'T', 'C', 'G']) for _ in range(size)])

def get_random_binary_seq(size):
    return ''.join([random.choice(['0', '1']) for _ in range(size)])

def get_positions(seq, pattern, initial_index):
    indexes = []
    while True:
        try:
            idx = seq.index(pattern, initial_index)
        except:
            break
        else:
            indexes.append(idx)
            initial_index = idx + 1
    return indexes

def get_align_text(aln):
    last_node = (0, 0)
    target = ''
    query = ''
    path = aln.__dict__.get('path', [(aln.coordinates[0, i], aln.coordinates[1, i]) for i in range(aln.coordinates.shape[1])])
    for node in path[1:]:
        steps_up = node[0] - last_node[0]
        steps_bm = node[1] - last_node[1]
        if steps_up == steps_bm:
            target += aln.target[last_node[0]:node[0]]
            query += aln.query[last_node[1]:node[1]]
        elif steps_bm == 0:
            target += aln.target[last_node[0]:node[0]]
            query += '-' * steps_up
        elif steps_up == 0:
            target += '-' * steps_bm
            query += aln.query[last_node[1]:node[1]]
        last_node = node
    return target, query

class AlignResult:

    @property
    def path(self):
        self._aln.__dict__.get('path', [(self._aln.coordinates[0, i], self._aln.coordinates[1, i]) for i in range(self._aln.coordinates.shape[1])])
    
    @property
    def coordinates(self):
        return self._aln.coordinates

    @property
    def counts(self):
        return self._aln.counts

    @property
    def frequencies(self):
        return self._aln.frequencies

    @property
    def score(self):
        return self._aln.score
        
    @property
    def sequences(self):
        return self._aln.sequences
    
    @property
    def substitutions(self):
        return self._aln.substitutions

    @property
    def query(self):
        return self._aln.query

    @property
    def target(self):
        return self._aln.target
    
    @property
    def aligned(self):
        return self._aln.aligned
    
    @property
    def seqA(self):
        return self._seqA

    @property
    def seqB(self):
        return self._seqB

    def __repr__(self):
        return f'{f"score {self.score}".center(40, "-")}\ntarget {self.seqA}\nquery  {self.seqB}'
    
    def __init__(self, aln):
        self._aln = aln
        self._seqA, self._seqB = get_align_text(self._aln)

class Aligner:
    """
    parameters for pair-wise sequence alignment
    """
    def __init__(self, match=2, mismatch=-1, gap_open=-0.5, gap_extension=-0.1):
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = match
        aligner.mismatch_score = mismatch
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extension
        aligner.target_end_gap_score = 0.0
        aligner.query_end_gap_score = 0.0
        self.aligner = aligner
    
    def __repr__(self):
        return str(self.aligner)
    
    def __call__(self, target, query):
        alns = self.aligner.align(seqA=target, seqB=query)
        rz = []
        num = 0
        for aln in alns:
            num += 1
            rz.append(AlignResult(aln))
            if num > 10:
                break
        return [AlignResult(aln) for aln in sorted(alns, key=lambda x: x.score, reverse=True)[:10]]

def add_rs_info(rs, src: str, ebi_size, rs_item_size):
    bit_segments = [int(src[i:i+rs_item_size], 2) for i in range(0, ebi_size, rs_item_size)]
    bit_segments = rs.encode(bit_segments)
    bits = ''.join([format(bseg, f'0{rs_item_size}b') for bseg in bit_segments])
    return bits

def remove_rs_info(rs, bits, rs_item_size, binary_length):
    bit_segments = [int(bits[i:i+rs_item_size], 2) for i in range(0, binary_length, rs_item_size)]
    try:
        bit_segs, _ ,_ = rs.decode(bit_segments)
    except:
        return False
    else:
        bits = ''.join([format(bseg, f'0{rs_item_size}b') for bseg in bit_segs])
        return bits

def get_randomfactors(rng, midx, segment_num, ebi_segment_size):
    random_factors = []
    for i in range(segment_num):
        rng.seed(midx + i+ 1)
        random_factors.append(''.join(rng.choice(ebi_segment_size, options=['0', '1'])))
    return random_factors
