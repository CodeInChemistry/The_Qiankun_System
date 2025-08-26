#!/usr/bin/env python3

"""
Script Name: simulation.py
Version: 1.0.1
Author: Shihong CHEN<shihong.chen@connect.ust.hk>
Date: 2025-06-23
"""

import random
import numpy as np
import Levenshtein as Lev

# introduce mutation to dna molecules set
def modifyArecord(dna, m_type, pos, m_dict):
    if m_type == 1: # deletion
        dna2 = dna[:pos] + '-' + dna[pos+1:]
        if pos in m_dict:
            m_dict[pos] = (m_dict[pos][0], '-')
        else:
            m_dict[pos] = (dna[pos], '-')
    elif m_type == 2: # insertion
        b_idx = random.choice([0, 1, 2, 3])
        add_base = 'ATCG'[b_idx]
        dna2 = dna[:pos] + add_base + dna[pos:]
        if f'i{pos}' in m_dict:
            m_dict[f'i{pos}'] = ('-', add_base+m_dict[pos][1])
        else:
            m_dict[f'i{pos}'] = ('-', add_base)
    elif m_type == 3: # substitution
        b_idx = random.choice([0, 1, 2, 3])
        if pos in m_dict:  
            if m_dict[pos][0] == 'ATCG'[b_idx]:
                b_idx -= 1
            nbase = 'ATCG'[b_idx]
            dna2 = dna[:pos] + nbase + dna[pos+1:]
            m_dict[pos] = (m_dict[pos][0], nbase)
        else:
            if dna[pos] == 'ATCG'[b_idx]:
                b_idx -= 1
            nbase = 'ATCG'[b_idx]
            dna2 = dna[:pos] + nbase + dna[pos+1:]
            m_dict[pos] = (dna[pos], nbase)
    return dna2, m_dict

def getArecord(dna, modifications):
    
    dna2 = dna[:]
    mut_dict = {}
    for error in modifications:
        dna2, mut_dict = modifyArecord(dna2, error[1], error[0], mut_dict)
    
    operations = ''
    for kw in mut_dict:
        if isinstance(kw, str):
            operations = f'{mut_dict[kw][0]}{int(kw[1:])+1}{mut_dict[kw][1]};' + operations
        else:
            operations = f'{mut_dict[kw][0]}{kw+1}{mut_dict[kw][1]};' + operations

    d2 = dna2.replace('-', '')
    return d2, operations, Lev.distance(dna, d2)

def simulate_reading(dna, copy_number=1, error_rate=[0.005, 0.001, 0.003], 
                       error_nums=[0, 0, 0], calculate_on_error_rate=False):
    
    """
    simulate reading process of DNA storage
    
    Parameters:
    dna (str): target DNA molecule sequence.
    copy_number (int): reading times for the DNA
    error_rate (list[float]): error rate for deletion, insertion, substitution (error bases / total bases).
    error_nums (list[int]): error num for each type of errors, deletion, insertion, substitution respectively.
    calculate_on_error_rate (bool): generate error num by the error rate or use the specific error_nums.

    Returns:
    list[(str, int)]: return a list of reading results defined as a tuple with mutated DNA and distance to original DNA.
    """
    
    rz = []
    error_nums = []
    size = len(dna)
    for _ in range(copy_number):
        errors = []
        if calculate_on_error_rate:
            for i in range(size):
                if random.random() < error_rate[0]:
                    errors.append((i, 1))
                if random.random() < error_rate[1]:
                    errors.append((i, 2))
                if random.random() < error_rate[2]:
                    errors.append((i, 3))
        else:
            for i in range(3):
                for j in range(error_nums[i]):
                    errors.append((random.randint(0, size-1), i+1))
                    
        errors_sorted = sorted(errors, key=lambda x: (x[0], x[1]), reverse=True)
        molecule, _, dis = getArecord(dna, errors_sorted)
        
        rz.append((molecule, dis))
    return rz
