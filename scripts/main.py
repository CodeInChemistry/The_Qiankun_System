#!/usr/bin/env python3

"""
Script Name: main.py
Version: 1.0.1
Author: Shihong CHEN<shihong.chen@connect.ust.hk>
Date: 2025-06-23
"""
import os, sys
from QianKun import QianKunAdvance
import matplotlib.pyplot as plt
import configparser
import argparse
import json

def get_dna_set(in_file):
    dna_set = []
    fhd = open(in_file)
    for line in fhd:
        line = line.strip('\n\r')
        dna_set.append(line)
    fhd.close()
    return dna_set

def save_dna_set(dna_set, output_file):
    fhd = open(output_file, 'w')
    for dna in dna_set:
        fhd.write(f'{dna}\n')
    fhd.close()

def get_params():
    # get arguments and parameters from command line
    parser = get_parser()
    args = parser.parse_args()
    
    operation = args.operation
    in_file = os.path.abspath(args.input_file)
    out_file = os.path.abspath(args.output_file)
    bianry_file_bit_num = None
    if not (os.path.exists(in_file) and os.access(in_file, os.R_OK)):
        raise Exception(f'input file {in_file} is not exists or is not accessible')
    
    file_idx = 1
    if os.access(os.path.dirname(out_file), os.W_OK):
        if os.path.exists(out_file):
            sys.stderr.write(f'output file {out_file} is exists, replace it (R) or keep both (K)\n')
            confirm = input(f'input your selection [R, K]: ')
            if confirm in ['R', 'r']:
                pass
            elif confirm in ['K', 'k']:
                p1, p2 = os.path.splitext(out_file)
                while 1:
                    tmp_path = f'{p1}.{file_idx}{p2}'
                    if not os.path.exists(tmp_path):
                        out_file = tmp_path
                        break
                    else:
                        file_idx += 1
            else:
                raise Exception('Unknown operation\n')
    else:
        raise Exception(f'output file {out_file} is not in a writable path\n')
    
    # get default arguments from default config file or specific config file
    if args.config_file:
        config_file = args.config_file
    else:
        cdir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(cdir, 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file)
    # QK Params
    hamming_code_sys=config.get('QianKunParams', 'hamming_code_sys')
    segment_num=config.getint('QianKunParams', 'segment_num')
    batch_size=config.getint('QianKunParams', 'batch_size')
    segment_diff=config.getint('QianKunParams', 'segment_diff')
    max_indel=config.getint('QianKunParams', 'max_indel')
    pseudobase=config.get('QianKunParams', 'pseudobase')
    rs_check_key=config.getint('QianKunParams', 'rs_check_key')
    encryption=config.getboolean('QianKunParams', 'encryption')
    # Aligner Params
    match=config.getfloat('AlignerParams', 'match')
    mismatch=config.getfloat('AlignerParams', 'mismatch')
    gap_open=config.getfloat('AlignerParams', 'gap_open')
    gap_extension=config.getfloat('AlignerParams', 'gap_extension')
    # IO Params
    file_key=config.getint('IOParams', 'file_key')
    
    # update the parameters if command line provide specific value
    if args.hamming_code_sys:
        hamming_code_sys = args.hamming_code_sys
    if args.segment_num:
        segment_num = args.segment_num
    if args.batch_size:
        batch_size = args.batch_size
    if args.segment_diff:
        segment_diff = args.segment_diff
    if args.max_indel:
        max_indel = args.max_indel
    if args.pseudobase:
        pseudobase = args.pseudobase
    if args.rs_check_key:
        rs_check_key = args.rs_check_key
    if args.add_encryption:
        encryption = args.add_encryption
    if args.match:
        match = args.match
    if args.mismatch:
        mismatch = args.mismatch
    if args.gap_open:
        gap_open = args.gap_open
    if args.gap_extension:
        gap_extension = args.gap_extension
    if args.bianry_file_bit_num:
        bianry_file_bit_num = args.bianry_file_bit_num
    if args.file_key:
        file_key = args.file_key
    
    return operation, in_file, out_file, bianry_file_bit_num, hamming_code_sys, segment_num, batch_size, segment_diff, max_indel, pseudobase, rs_check_key, encryption, match, mismatch, gap_open, gap_extension, file_key
    
    
def main():
    
    operation, in_file, out_file, bianry_file_bit_num, hamming_code_sys, segment_num, batch_size, segment_diff, max_indel, pseudobase, rs_check_key, encryption, match, mismatch, gap_open, gap_extension, file_key = get_params()
    
    # construct QKA object with the parameters
    aligner = Aligner(match=match, mismatch=mismatch, gap_open=gap_open,  gap_extension=gap_extension)
    qk = QianKunAdvance(
                 yyc_code_index=[873, 918], 
                 hamming_code_sys=hamming_code_sys, 
                 segment_num=segment_num, 
                 batch_size=batch_size, 
                 segment_diff=segment_diff, 
                 max_indel=max_indel, 
                 pseudobase=pseudobase, 
                 aligner=aligner, 
                 rs_check_key=rs_check_key,
                 randomize=encryption
                )
    
    # output log information to stdout
    print('Task Description'.center(80, '='))
    print(f'Operation: {operation}')
    print(f'Input file: {in_file}')
    print(f'Output file: {out_file}')
    print(f'Encryption key: {file_key}')
    print(f'Binary information size: {bianry_file_bit_num}')
    print(f'Qiankun Params: {qk}')
    print(''.center(80, '='))
    
    # Run the code to finish the operation
    if operation == 'encode':
        file_bit_size, DNA_molecules, binary_fragments = qk.encode(in_file)
        save_dna_set(DNA_molecules, out_file)
    elif operation == 'decode':
        dna_set = get_dna_set(in_file)
        seg_dict, s_binary = qk.decode(DNA_set=dna_set, output_file=out_file, file_key=file_key, file_bit_size=bianry_file_bit_num)
    else:
        raise Exception(f'Unknown operation \'{operation}\'')
    print('Done')


if __name__ == '__main__':
    main()
