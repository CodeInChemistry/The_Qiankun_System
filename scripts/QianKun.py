#!/usr/bin/env python3

"""
Script Name: Qiankun.py
Version: 1.0.1
Author: Shihong CHEN<shihong.chen@connect.ust.hk>
Date: 2025-06-23
"""


from utilities import Aligner, YYC, HammingCode, WELL_N_RNG
from utilities import reverse_complemnet, add_rs_info, remove_rs_info
from utilities import decimal_to_factorial_num, ftnum_to_order_seq, order_seq_to_ftnum, factorial_num_to_decimal
from utilities import get_randomfactors, get_random_binary_seq, get_random_dna_seq, get_positions
from utilities import XOR, XOR2
from utilities import get_binary_info, save_binary_info, factorial
from simulation import simulate_reading

import re
import os
import sys
import itertools
import copy
import random
import time
import numpy as np
import reedsolo
import Levenshtein as Lev
import multiprocessing


def worker(func, queue, **kwargs):
    """Worker function that executes the given function with keyword arguments."""
    result = func(**kwargs)
    # rz = [(a, b, c, d, e, kwargs['mutated_dna']) for a, b, c, d, e in result]
    queue.put(result)  # Put the result in the queue
    
class QianKunAdvance:

    def __init__(self, fragment_num:int=10, yyc_codec_index: int=873, hamming_code_type:str='16-11', rs_item_num:int=0,
                 encryption:bool=True, batch_size:int=3, aligner=Aligner(), check_threshold:int=3, pseudobase:str='A', 
                 time_limit=None):
        # object parameters
        self.yyc = YYC(yyc_codec_index)
        self.hc = HammingCode(hamming_code_type)
        self.rng = WELL_N_RNG()
        self.encryption = encryption
        self.fragment_num = fragment_num
        self.rs_item_num = rs_item_num
        self.batch_size = batch_size
        self.timeout = time_limit
        self.check_threshold = check_threshold
        self.aligner = aligner
        self.pseudobase = pseudobase
        
        # variables needed in encoding and decoding
        self.ebi_segment_size = self.hc.src_size
        self.dna_segment_size = self.hc.tgt_size // 2
        self.ebi_size = self.ebi_segment_size * self.fragment_num
        self.pure_ebi_size = self.ebi_size - self.rs_item_num * self.ebi_segment_size
        if self.rs_item_num == 0:
            self.rs = None
        else:
            self.rs = reedsolo.RSCodec(nsym=self.rs_item_num, nsize=self.fragment_num, c_exp=self.ebi_segment_size, prim=285, generator=2, single_gen=True, fcr=1)

        # parameters for decoding
        self.safe_island_threshold = 2

    def add_rs_info(self, src):
        src_items = [int(src[i:i+self.ebi_segment_size], 2) for i in range(0, self.pure_ebi_size, self.ebi_segment_size)]
        tgt_items = self.rs.encode(src_items)
        tgt = ''.join([format(item, f'0{self.ebi_segment_size}b') for item in tgt_items])
        return tgt

    def remove_rs_info(self, tgt, error_pos=None):
        tgt_items = [int(tgt[i:i+self.ebi_segment_size], 2) for i in range(0, self.ebi_size, self.ebi_segment_size)]
        try:
            src_items, _ ,_ = self.rs.decode(tgt_items, erase_pos=error_pos)
        except:
            return False
        else:
            src = ''.join([format(item, f'0{self.ebi_segment_size}b') for item in src_items])
            return src
    
    def get_randomfactors(self, midx):
        random_factors = []
        for i in range(self.fragment_num):
            self.rng.seed(midx + i+ 1)
            random_factors.append(''.join(self.rng.choice(self.ebi_segment_size, options=['0', '1'])))
        return random_factors
    
    def get_DNA_molecule(self, bis, midx):
        
        assert len(bis) <= self.pure_ebi_size, f"bis should not be longer than {self.pure_ebi_size}\n"
        
        # check binary infomation length, standardize it by filling random binary info
        if len(bis) != self.pure_ebi_size:
            bis += get_random_binary_seq(self.pure_ebi_size - len(bis))

        # add rs items
        if self.rs is not None:
            bis = self.add_rs_info(bis)

        # do encryption
        if self.encryption:
            random_info = ''.join(self.get_randomfactors(midx))
            bis = XOR2([bis, random_info])
        
        # generate dna fragments
        segments = []
        for bidx in range(0, self.ebi_size, self.ebi_segment_size):
            binfo_hm = self.hc.encode(bis[bidx:bidx+self.ebi_segment_size])
            dna_segment = self.yyc.encode(binfo_hm[:self.dna_segment_size], binfo_hm[self.dna_segment_size:])
            segments.append(dna_segment)
    
        # construct a DNA molecule
        ftnum = decimal_to_factorial_num(midx, self.fragment_num)
        frag_order_seq = ftnum_to_order_seq(ftnum)
        dna_molecule = ''.join(segments + [reverse_complemnet(segments[i]) for i in frag_order_seq])
        return dna_molecule

    def encode(self, input_file:str, file_key=1812433257, output_file:str=None):
        
        time_stamp_start = time.asctime()
        self.rng.update(prime=file_key)
        
        bs = get_binary_info(input_file)
        file_bit_size = len(bs)
        binary_fragments = []
        DNA_molecules = []
        
        midx = 0
        batch = []
        miss_num = 0
        miss_idx = []
        
        for idx in range(0, file_bit_size, self.pure_ebi_size):
            # log encoding progress
            sys.stderr.write(f'\rEncoding: {idx /file_bit_size*100:.2f} %')

            # get binary fragment
            binary_fragment = bs[idx:idx+self.pure_ebi_size]
            if len(binary_fragment) < self.pure_ebi_size:
                binary_fragment += get_random_binary_seq(self.pure_ebi_size-len(binary_fragment))

            binary_fragments.append(binary_fragment)
            batch.append(binary_fragment)
            dna = self.get_DNA_molecule(bis=binary_fragment, midx=midx)
            DNA_molecules.append(dna)
            
            # check if the batch is full
            if len(batch) == self.batch_size:
                midx += 1
                xor_info = XOR2(batch)
                binary_fragments.append(xor_info)
                edna = self.get_DNA_molecule(bis=xor_info, midx=midx)
                DNA_molecules.append(edna)
                batch = []
            midx += 1
            
        if len(batch) > 0:
            # if the last batch is full
            if len(batch) < self.batch_size:
                # fill the batch with random information
                random_binary_fragment_num = self.batch_size - len(batch)
                for _ in range(random_binary_fragment_num):
                    bis = get_random_binary_seq(self.pure_ebi_size)
                    binary_fragments.append(bis)
                    batch.append(bis)
                    dna = self.get_DNA_molecule(bis=bis, midx=midx)
                    DNA_molecules.append(dna)
                    midx += 1
            # dispose the last batch
            xor_info = XOR2(batch)
            binary_fragments.append(xor_info)
            edna = self.get_DNA_molecule(bis=xor_info, midx=midx)
            DNA_molecules.append(edna)
            batch = []
        sys.stderr.write(f'\rEncoding: {100:.2f} %')
        time_stamp_end = time.asctime()
        if output_file is not None:
            fout = open(output_file, 'w')
            # output file header
            fout.write(f'#'+''.center(80, '=')+'\n')
            fout.write(f'# Codec System: YYC({self.yyc.code_sys_idx})\n')
            fout.write(f'# Hamming Code: {self.hc.tgt_size}-{self.hc.src_size}\n')
            fout.write(f'# RS Code item: {self.rs_item_num}\n')
            fout.write(f'# Fragment num: {self.fragment_num}\n')
            fout.write(f'# Encryption: {self.encryption}\n')
            fout.write(f'# Batch size: {self.batch_size}\n')
            fout.write(f'# File key: {file_key}\n')
            fout.write(f'# input file: {input_file}\n')
            fout.write(f'# output file: {output_file}\n')
            fout.write(f'# Started at: {time_stamp_start}\n')
            fout.write(f'# Finished at: {time_stamp_end}\n')
            fout.write(f'#'+''.center(80, '=')+'\n')
            for dna in DNA_molecules:
                fout.write(f'{dna}\n')
            fout.close()
        return file_bit_size, DNA_molecules, binary_fragments

    def get_decode_status(self, seg):
        # Is a substitution in a pair, check the decode status for a segment
        ia1, ib1 = self.yyc.decode(seg, pseudobase=self.pseudobase)
        bis = ia1+ib1
        if len(bis) != self.hc.tgt_size:
            return -1, None
        s1, s2 = self.hc.decode(ia1+ib1)
        if s2 == 0:
            return 10, s1
        elif s2 > 0:
            return 1, s1
        else:
            return -1, None
        
    def get_fragments(self, d_seq):
        forward_fragments = []
        reverse_fragments = []
        for i in range(0, len(d_seq)-self.dna_segment_size+1):
            fragment = d_seq[i:i+self.dna_segment_size]
            rc_fragment = reverse_complemnet(fragment)
            
            # s1 = get_decode_status(fragment, self.yyc, self.hc, self.pseudobase)
            # s2 = get_decode_status(rc_fragment, self.yyc, self.hc, self.pseudobase)
            s1 = self.get_decode_status(fragment)
            s2 = self.get_decode_status(rc_fragment)
    
            if s1[0] in [10]:
                forward_fragments.append((i, i+self.dna_segment_size, fragment, self.dna_segment_size, s1[0], s1[1], 'f'))
            if s2[0] in [10]:
                reverse_fragments.append((i, i+self.dna_segment_size, rc_fragment, self.dna_segment_size, s2[0], s2[1], 'r'))
        return forward_fragments + reverse_fragments

    def merge_fragments(self, fragments, molecule_size, safe_island_threshold=2):
        merged_fragments = []
        start_idx = 0
        selected_idx = [0]
        while 1:
            merged_fragment = [fragments[start_idx][0], fragments[start_idx][1], [start_idx], fragments[start_idx][1] - fragments[start_idx][0], fragments[start_idx][-1]]
                
            for i in range(start_idx+1, len(fragments)):
                if merged_fragment[1] == fragments[i][0]:
                    if merged_fragment[-1] == fragments[i][-1]:
                        merged_fragment[1] = fragments[i][1]
                        merged_fragment[2].append(i)
                        merged_fragment[3] += fragments[i][1] - fragments[i][0]
                        selected_idx.append(i)
                elif merged_fragment[1] < fragments[i][0]:
                    break
            merged_fragments.append(merged_fragment)
            
            # get a new start idx
            while 1:
                start_idx += 1
                if start_idx not in selected_idx:
                    break
            if start_idx >= len(fragments):
                break
    
        # validate merged fragments
        final_mf = []
        for mf in merged_fragments:
            if len(mf[2]) < safe_island_threshold:
                if mf[0] == 0 or mf[1] == molecule_size:
                    final_mf.append(mf)
            elif len(mf[2]) > self.fragment_num:
                if mf[-1] == 'f':
                    f_list = mf[2][:self.fragment_num]
                    lpos = mf[0]
                    rpos = fragments[f_list[-1]][1]
                    size = rpos-lpos
                else:
                    f_list = mf[2][-self.fragment_num:]
                    lpos = fragments[f_list[0]][0]
                    rpos = mf[1]
                    size = rpos-lpos
                final_mf.append([lpos, rpos, f_list, size, mf[-1]])
            else:
                final_mf.append(mf)
    
        # remove the less likely merged fragment
        # if f[0] <= f'[0] < f'[1] <= f[1], remove f'
        sorted_fmf = sorted(final_mf, key=lambda x: x[3], reverse=True)
        final_result = []
        for f in sorted_fmf:
            remove = False
            for f2 in final_result:
                if f2[0] <= f[0] < f[1] <= f2[1]:
                    remove = True
                    break
            if not remove:
                final_result.append(f)
        for f in final_result:
            weight = len(f[2])
            if f[0] == 0 or f[1] == molecule_size:
                weight += 1
            f.append(weight)
        
        return sorted(final_result, key=lambda x: x[-1], reverse=True)

    def dispose_gaps(self, new_map, mutated_dna, time_stamp=None):
        d_seq_len = len(mutated_dna)
        start_pos = 0
        gaps = []
        for fragment in new_map:
            if start_pos < fragment[0]:
                gap_size = fragment[0] - start_pos
                gf_num = np.round(gap_size / self.dna_segment_size)
                indel_num = np.abs(gf_num * self.dna_segment_size - gap_size)
                gaps.append([start_pos, fragment[0], '', gap_size, -1, '', 'g', gf_num, indel_num])
            start_pos = fragment[1]
        if start_pos < d_seq_len:
            gap_size = d_seq_len - start_pos
            gf_num = np.round(gap_size / self.dna_segment_size)
            indel_num = np.abs(gf_num * self.dna_segment_size - gap_size)
            gaps.append([start_pos, d_seq_len, '', gap_size, -1, '', 'g', gf_num, indel_num])
        
        delta_f_num = int(len(new_map) + np.sum([f[-2] for f in gaps])) - self.fragment_num * 2

        if delta_f_num != 0 and len(gaps) == 0:
            return []
            
        while delta_f_num != 0:
            if delta_f_num > 0:
                # make gap fragment less
                modify_idx = np.argmin([np.abs(f[3] - (f[7]-1) * self.dna_segment_size) - f[-1] for f in gaps])
                gaps[modify_idx][7] -= 1
                gaps[modify_idx][8] = np.abs(gaps[modify_idx][3] - gaps[modify_idx][7] * self.dna_segment_size)
                delta_f_num -= 1
            else:
                # make gap fragment more
                modify_idx = np.argmin([np.abs(f[3] - (f[7]+1) * self.dna_segment_size) - f[-1] for f in gaps])
                gaps[modify_idx][7] += 1
                gaps[modify_idx][8] = np.abs(gaps[modify_idx][3] - gaps[modify_idx][7] * self.dna_segment_size)
                delta_f_num += 1
        
        # check gaps that can be merged
        merge_gaps_options = []
        for i in range(1, len(gaps)):
            tgt_indel = np.abs(gaps[i][3] + gaps[i-1][3] - (gaps[i][7] + gaps[i-1][7]) * self.dna_segment_size)
            src_indel = gaps[i][8] + gaps[i-1][8]
            island_size = (gaps[i][0] - gaps[i-1][1]) // self.dna_segment_size
            if tgt_indel < src_indel and island_size < src_indel - tgt_indel:
                merge_gap = [gaps[i-1][0], gaps[i][1], '', gaps[i][1]-gaps[i-1][0], -1, '', 
                             'g', gaps[i][7] + gaps[i-1][7] + island_size, tgt_indel, src_indel - tgt_indel, island_size]
                merge_gaps_options.append((i-1, i, src_indel, tgt_indel, merge_gap))
        
        # check if conflict between merged gaps
        merged_gaps = []
        if len(merge_gaps_options) <=1:
            merged_gaps = [mg[-1][:9] for mg in merge_gaps_options]
        else:
            mg_idx = 0
            while mg_idx < len(merge_gaps_options)-1:
                if  merge_gaps_options[mg_idx+1][0] == merge_gaps_options[mg_idx][1]:
                    if merge_gaps_options[mg_idx+1][-1][-1] > merge_gaps_options[mg_idx][-1][-1]:
                        # keep left
                        merged_gaps.append(merge_gaps_options[mg_idx][-1][:9])
                        mg_idx += 2
                    elif merge_gaps_options[mg_idx+1][-1][-1] == merge_gaps_options[mg_idx][-1][-1] and merge_gaps_options[mg_idx+1][-2] < merge_gaps_options[mg_idx][-2]:
                        # keep left
                        merged_gaps.append(merge_gaps_options[mg_idx][-1][:9])
                        mg_idx += 2
                    else:
                        # keep right
                        mg_idx += 1
                else:
                    merged_gaps.append(merge_gaps_options[mg_idx][-1][:9])
                    mg_idx += 1
            if mg_idx == len(merge_gaps_options)-1:
                merged_gaps.append(merge_gaps_options[mg_idx][-1][:9])

        # update the map
        updated_map = []
        for seg in new_map:
            drop = False
            for mg in merged_gaps:
                if mg[0] < seg[0] < mg[1]:
                    drop = True
                    break
            if not drop:
                updated_map.append(seg)
        single_gaps = []
        for seg in gaps:
            drop = False
            for mg in merged_gaps:
                if mg[0] <= seg[0] < mg[1]:
                    drop = True
                    break
            if not drop:
                single_gaps.append(seg)
        gap_options = []
        final_gaps = sorted(merged_gaps+single_gaps, key=lambda x: x[0])
        
        for gap in final_gaps:

            gap_fragment_num = int(gap[7])
            if gap_fragment_num == 0:
                continue
            gap_size = int(gap[3] // gap_fragment_num)
            residual_size = -gap_fragment_num * gap_size + gap[3]
            gap_fragment_sizes = list(itertools.combinations(range(gap_fragment_num), residual_size))

            g_options = []
            for gfs in gap_fragment_sizes:
                gap_frag_size = [gap_size for _ in range(gap_fragment_num)]
                for gfidx in gfs:
                    gap_frag_size[gfidx] += 1

                new_gaps = []
                start_pos = gap[0]
                for gsize in gap_frag_size:
                    end_pos = start_pos + gsize
                    new_gaps.append([start_pos, end_pos, '', end_pos-start_pos, -1, '', 'g'])
                    start_pos = end_pos
                g_options.append(new_gaps)
            gap_options.append(g_options)
        gap_sets = [[]]
        for g_options in gap_options:
            
            if self.timeout is not None:
                if time.time() - time_stamp > self.timeout:
                    break
                    
            updated_gap_sets = []
            for gap_opt in g_options:
                tmp_gap_sets = copy.deepcopy(gap_sets)
                for gap_set in tmp_gap_sets:
                    gap_set += gap_opt
                updated_gap_sets += tmp_gap_sets
            gap_sets = updated_gap_sets
        return [sorted(copy.deepcopy(updated_map) + copy.deepcopy(gap_set), key=lambda x:x[0]) for gap_set in gap_sets]
    
    def getMaps(self, mutated_dna, time_stamp=None):
        d_seq_len = len(mutated_dna)
        
        fragments = self.get_fragments(d_seq=mutated_dna)
        
        merged_fragments = self.merge_fragments(fragments, d_seq_len, self.safe_island_threshold)

        the_map = []
        for mfidx, mf in enumerate(merged_fragments):
            conflicts = []
            new_feature = copy.deepcopy(mf)
            # [start, end, [fidxes], size, direction, weigtht]
            for mf2 in the_map:
                # check direction
                if new_feature[0] < mf2[0]:
                    if new_feature[4] == 'r' and mf2[4] == 'f':
                        new_feature[2] = []
                        break
                if new_feature[0] > mf2[0]:
                    if new_feature[4] == 'f' and mf2[4] == 'r':
                        new_feature[2] = []
                        break
                if new_feature[0] <= mf2[0] < new_feature[1]:
                    new_feature[2] = [fidx for fidx in new_feature[2] if fragments[fidx][1] < mf2[0]]
                    if len(new_feature[2]) == 0:
                        break
                    new_feature[1] = fragments[new_feature[2][-1]][1]
                    new_feature[3] = new_feature[1] - new_feature[0]
                    new_feature[5] = len(new_feature[2])
                    if new_feature[0] == 0 or new_feature[1] == d_seq_len:
                        new_feature[5] += 1
                elif mf2[0] <= new_feature[0] < mf2[1]:
                    new_feature[2] = [fidx for fidx in new_feature[2] if fragments[fidx][0] >= mf2[1]]
                    if len(new_feature[2]) == 0:
                        break
                    new_feature[1] = fragments[new_feature[2][-1]][1]
                    new_feature[3] = new_feature[1] - new_feature[0]
                    new_feature[5] = len(new_feature[2])
                    if new_feature[0] == 0 or new_feature[1] == d_seq_len:
                        new_feature[5] += 1
            if new_feature[-1] >= self.safe_island_threshold:
                the_map.append(new_feature)

        # reformat the map
        new_map = [list(fragments[fidx]) for mf in the_map for fidx in mf[2]]
        new_map = sorted(new_map, key=lambda x: x[0])

        
        # get gaps
        maps = self.dispose_gaps(new_map, mutated_dna, time_stamp)

        if len(maps) == 0:
            return fragments, merged_fragments, maps
        
        # check if we need to reset the gaps
        amap = copy.deepcopy(maps[0])
        fidx = 0
        reset_gaps = False
        # check the some fragment are not in correct direction
        for f in amap:
            if fidx < self.fragment_num:
                f += ['f', fidx, self.dna_segment_size - f[3]]
                f[2] = mutated_dna[f[0]:f[1]]
                if f[6] == 'r':
                    f[6] = 'g'
                    f[4] = -1
                    reset_gaps =True
            else:
                f += ['r', fidx]
                f[2] = reverse_complemnet(mutated_dna[f[0]:f[1]])
                if f[6] == 'f':
                    f[6] = 'g'
                    f[4] = -1
                    reset_gaps =True
            fidx +=1
            
        if reset_gaps:
            new_map = [f[:7] for f in amap if f[6] != 'g']
            maps = self.dispose_gaps(new_map, mutated_dna, time_stamp)
            # maps = [sorted(copy.deepcopy(new_map) + copy.deepcopy(gap_set), key=lambda x:x[0]) for gap_set in gap_sets]
            
        # add fragment index
        for amap in maps:
            
            fidx = 0
            reset_gaps = False
            # check the some fragment are not in correct direction
            for f in amap:
                if fidx < self.fragment_num:
                    f += ['f', fidx]
                    f[2] = mutated_dna[f[0]:f[1]]
                else:
                    f += ['r', fidx]
                    f[2] = reverse_complemnet(mutated_dna[f[0]:f[1]])
                fidx +=1
            
            # fill fragment to a map if a fragment fix the shape
            for f in amap:
                if f[6] == 'g' and (f[-1] == 0 or (f[0] == amap[f[-1]-1][1] and amap[f[-1]-1][6] != 'g')):
                    for fragment in fragments:
                        if (fragment[0], fragment[1]) == (f[0], f[1]) and fragment[6] == f[-2]:
                            amap[f[-1]] = list(fragment) + f[-2:]
                            break
                elif f[6] == 'g' and (f[-1] == self.fragment_num*2-1 or (f[1] == amap[f[-1]+1][0] and amap[f[-1]+1][6] != 'g')):
                    for fragment in fragments:
                        if (fragment[0], fragment[1]) == (f[0], f[1]) and fragment[6] == f[-2]:
                            amap[f[-1]] = list(fragment) + f[-2:]
                            break
            
        return fragments, merged_fragments, maps

    def polish_solution(self, src_solution, d_seq):
        """
        traverse the solution fragments and check the edge of gap fragments
        if a better edge option is found, update the solution
        """
    
        # sort the solution by the starting position of the fragments
        solution = sorted(copy.deepcopy(src_solution), key=lambda x:x[0])

        island_map = {}
        block = []
        edge_fragments = []
        for i in range(self.fragment_num*2):
            if solution[i][6] != 'g' and (i==0 or solution[i][0] == solution[i-1][1]):
                block.append(i)
            else:
                if len(block) > 0:
                    for kw in block:
                        island_map[kw] = len(block)
                    edge_fragments += [block[0], block[-1]]
                if solution[i][6] != 'g':
                    block = [i]
                else:
                    block = []
        if len(block) > 0:
            for kw in block:
                island_map[kw] = len(block)
            edge_fragments += [block[0], block[-1]]
            block = []

        while 1:
            changed = False
            for sidx, seg in enumerate(solution):
                # if two error-free fragments match to each other and not identical, make them as gap fragment
                if seg[9] > 0 and seg[4] == 10 and solution[seg[10]][4] == 10:
                    seg[6] = 'g'
                    seg[4] = -1
                    solution[seg[-1]][6] = 'g'
                    solution[seg[-1]][4] = -1
                    changed = True
                # if an error-free fragment with a not identical match, make it as a gap fragment if it stands at the edge of a safe island
                elif seg[9] > 0 and seg[6] in ['f', 'r'] and (not (seg[8] == 0 or solution[seg[8]-1][6] != 'g') or  not (seg[8] == self.fragment_num*2-1 or solution[seg[8]+1][6] != 'g')):
                    if seg[8] in edge_fragments:
                        seg[6] = 'g'
                        seg[4] = -1
                    else:
                        seg[6] = 'x'
                        seg[4] = -1
                    changed = True
            if not changed:
                break

        # set isolated fragment as a gap fragment
        for idx, seg in enumerate(solution):
            if seg[6] in ['f', 'r']:
                isolation = [0,0]
                if idx > 0:
                    # check left side
                    if solution[idx-1][6] != 'g':
                        isolation[0] = 1
                if idx < self.fragment_num * 2 -1:
                    if solution[idx+1][6] != 'g':
                        isolation[1] = 1
                
                if np.sum(isolation) == 0:
                    seg[6] = 'g'
                    seg[4] = -1

        # optimize gap fragment matching to non-gap fragment
        candidate_fragments = []
        for seg in solution:
            if seg[-2] > 0:
                if seg[6] == 'g' and solution[seg[-1]][6] != 'g':
                    if seg[-3] not in candidate_fragments:
                        candidate_fragments.append(seg[-3])
                if seg[6] in ['f', 'r'] and solution[seg[-1]][6] == 'g':
                    if seg[-1] not in candidate_fragments:
                        candidate_fragments.append(seg[-1])

        for i in candidate_fragments:
             while 1:
                changed = False
                 # optimize the left edge
                if i > 0:
                    if solution[i][0] > solution[i-1][1]:
                        tmp_l, tmp_r = solution[i][0]-1, solution[i][1]
                        p1 = solution[i][-1]
                        dnaf1 = d_seq[tmp_l:tmp_r]
                        if solution[i][7] == 'r':
                            dnaf1 = reverse_complemnet(dnaf1)
                        d1 = Lev.distance(dnaf1, solution[p1][2])
                        if d1 < solution[i][-2]:
                            # update solution
                            solution[i][0], solution[i][1] = tmp_l, tmp_r
                            solution[i][2] = dnaf1
                            solution[i][3] = tmp_r - tmp_l
                            solution[i][-2], solution[p1][-2] = d1, d1
                            changed = True
                            continue
                    elif solution[i-1][6] == 'g':
                        neibour_idx = i-1
                        tmp_l1, tmp_l2 = solution[i][0]-1, solution[neibour_idx][0]
                        tmp_r1, tmp_r2 = solution[i][1], solution[neibour_idx][1]-1
                        p1 = solution[i][-1]
                        p2 = solution[neibour_idx][-1]
                        dnaf1 = d_seq[tmp_l1:tmp_r1]
                        dnaf2 = d_seq[tmp_l2:tmp_r2]
                        if solution[i][7] == 'r':
                            dnaf1 = reverse_complemnet(dnaf1)
                        if solution[neibour_idx][7] == 'r':
                            dnaf2 = reverse_complemnet(dnaf2)
                        d1 = Lev.distance(dnaf1, solution[p1][2])
                        d2 = Lev.distance(dnaf2, solution[p2][2])
                        if d1 < solution[i][-2]:
                            solution[i][0], solution[neibour_idx][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[neibour_idx][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[neibour_idx][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[neibour_idx][-2] = d1, d2
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[i][2], solution[neibour_idx][2] = dnaf1, dnaf2
                            changed = True
                            continue
                        tmp_l1, tmp_l2 = solution[i][0]+1, solution[neibour_idx][0]
                        tmp_r1, tmp_r2 = solution[i][1], solution[neibour_idx][1]+1
                        p1 = solution[i][-1]
                        p2 = solution[neibour_idx][-1]
                        dnaf1 = d_seq[tmp_l1:tmp_r1]
                        dnaf2 = d_seq[tmp_l2:tmp_r2]
                        if solution[i][7] == 'r':
                            dnaf1 = reverse_complemnet(dnaf1)
                        if solution[neibour_idx][7] == 'r':
                            dnaf2 = reverse_complemnet(dnaf2)
                        d1 = Lev.distance(dnaf1, solution[p1][2])
                        d2 = Lev.distance(dnaf2, solution[p2][2])
                        if d1 < solution[i][-2]:
                            solution[i][0], solution[neibour_idx][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[neibour_idx][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[neibour_idx][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[neibour_idx][-2] = d1, d2
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[i][2], solution[neibour_idx][2] = dnaf1, dnaf2
                            changed = True
                            continue
                elif solution[i][0] > 0:
                        tmp_l, tmp_r = solution[i][0]-1, solution[i][1]
                        p1 = solution[i][-1]
                        dnaf1 = d_seq[tmp_l:tmp_r]
                        if solution[i][7] == 'r':
                            dnaf1 = reverse_complemnet(dnaf1)
                        d1 = Lev.distance(dnaf1, solution[p1][2])
                        if d1 < solution[i][-2]:
                            # update solution
                            solution[i][0], solution[i][1] = tmp_l, tmp_r
                            solution[i][2] = dnaf1
                            solution[i][3] = tmp_r - tmp_l
                            solution[i][-2], solution[p1][-2] = d1, d1
                            changed = True
                            continue 
                # optimize the right edge
                if i < self.fragment_num * 2 -1:
                    if solution[i][1] < solution[i+1][0]:
                        tmp_l, tmp_r = solution[i][0], solution[i][1]+1
                        p1 = solution[i][-1]
                        dnaf1 = d_seq[tmp_l:tmp_r]
                        if solution[i][7] == 'r':
                            dnaf1 = reverse_complemnet(dnaf1)
                        d1 = Lev.distance(dnaf1, solution[p1][2])
                        if d1 < solution[i][-2]:
                            # update solution
                            solution[i][0], solution[i][1] = tmp_l, tmp_r
                            solution[i][2] = dnaf1
                            solution[i][3] = tmp_r - tmp_l
                            solution[i][-2], solution[p1][-2] = d1, d1
                            changed = True
                            continue
                    elif solution[i+1][6] == 'g':
                        neibour_idx = i+1
                        tmp_l1, tmp_l2 = solution[i][0], solution[neibour_idx][0]-1
                        tmp_r1, tmp_r2 = solution[i][1]-1, solution[neibour_idx][1]
                        p1 = solution[i][-1]
                        p2 = solution[neibour_idx][-1]
                        dnaf1 = d_seq[tmp_l1:tmp_r1]
                        dnaf2 = d_seq[tmp_l2:tmp_r2]
                        if solution[i][7] == 'r':
                            dnaf1 = reverse_complemnet(dnaf1)
                        if solution[neibour_idx][7] == 'r':
                            dnaf2 = reverse_complemnet(dnaf2)
                        d1 = Lev.distance(dnaf1, solution[p1][2])
                        d2 = Lev.distance(dnaf2, solution[p2][2])
                        if d1 < solution[i][-2]:
                            solution[i][0], solution[neibour_idx][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[neibour_idx][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[neibour_idx][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[neibour_idx][-2] = d1, d2
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[i][2], solution[neibour_idx][2] = dnaf1, dnaf2
                            changed = True
                            continue
                        tmp_l1, tmp_l2 = solution[i][0], solution[neibour_idx][0]+1
                        tmp_r1, tmp_r2 = solution[i][1]+1, solution[neibour_idx][1]
                        p1 = solution[i][-1]
                        p2 = solution[neibour_idx][-1]
                        dnaf1 = d_seq[tmp_l1:tmp_r1]
                        dnaf2 = d_seq[tmp_l2:tmp_r2]
                        if solution[i][7] == 'r':
                            dnaf1 = reverse_complemnet(dnaf1)
                        if solution[neibour_idx][7] == 'r':
                            dnaf2 = reverse_complemnet(dnaf2)
                        d1 = Lev.distance(dnaf1, solution[p1][2])
                        d2 = Lev.distance(dnaf2, solution[p2][2])
                        if d1 < solution[i][-2]:
                            solution[i][0], solution[neibour_idx][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[neibour_idx][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[neibour_idx][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[neibour_idx][-2] = d1, d2
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[i][2], solution[neibour_idx][2] = dnaf1, dnaf2
                            changed = True
                            continue
                elif solution[i][1] < len(d_seq):
                    tmp_l, tmp_r = solution[i][0], solution[i][1]+1
                    p1 = solution[i][-1]
                    dnaf1 = d_seq[tmp_l:tmp_r]
                    if solution[i][7] == 'r':
                        dnaf1 = reverse_complemnet(dnaf1)
                    d1 = Lev.distance(dnaf1, solution[p1][2])
                    if d1 < solution[i][-2]:
                        # update solution
                        solution[i][0], solution[i][1] = tmp_l, tmp_r
                        solution[i][2] = dnaf1
                        solution[i][3] = tmp_r - tmp_l
                        solution[i][-2], solution[p1][-2] = d1, d1
                        changed = True
                        break

                if not changed:
                    break
             solution[i][6] = 'x'
        
        # optimize gap-pairs
        f_num = len(solution)
        while 1:    
            changed = False
            for i in range(f_num-1):
                if solution[i][6] in ['g', 'x'] and solution[i][1] < solution[i+1][0]:
                    # try to extend the gap region on right side
                    tmp_l, tmp_r = solution[i][0], solution[i][1]+1
                    p1 = solution[i][-1]
                    dnaf1 = d_seq[tmp_l:tmp_r]
                    if solution[i][7] == 'r':
                        dnaf1 = reverse_complemnet(dnaf1)
                    d1 = Lev.distance(dnaf1, solution[p1][2])
                    if d1 < solution[i][-2]:
                        # update solution
                        solution[i][0], solution[i][1] = tmp_l, tmp_r
                        solution[i][2] = dnaf1
                        solution[i][3] = tmp_r - tmp_l
                        solution[i][-2], solution[p1][-2] = d1, d1
                        changed = True
                        break
                elif solution[i][6] in ['g', 'x'] and i > 0 and solution[i][0] > solution[i-1][1]:
                    # try to extend the gap region on left side
                    tmp_l, tmp_r = solution[i][0]-1, solution[i][1]
                    p1 = solution[i][-1]
                    dnaf1 = d_seq[tmp_l:tmp_r]
                    if solution[i][7] == 'r':
                        dnaf1 = reverse_complemnet(dnaf1)
                    d1 = Lev.distance(dnaf1, solution[p1][2])
                    if d1 < solution[i][-2]:
                        # update solution
                        solution[i][0], solution[i][1] = tmp_l, tmp_r
                        solution[i][2] = dnaf1
                        solution[i][3] = tmp_r - tmp_l
                        solution[i][-2], solution[p1][-2] = d1, d1
                        changed = True
                        break
                elif (solution[i][6] in ['g', 'x'] and solution[i+1][6] in ['g', 'x']) or (solution[i][6] == 'x' and solution[i+1][6] == 'g'):
                    p1 = solution[i][-1]
                    p2 = solution[i+1][-1]

                    # try to make a two-step move
                    if solution[p1][6] in ['g', 'x'] and solution[p2][6] in ['g', 'x'] and np.abs(p1 - p2) == 1:
                        indel_src1 = np.abs(self.dna_segment_size - solution[i][3])
                        indel_src2 = np.abs(self.dna_segment_size - solution[i+1][3])
                        indel_src3 = np.abs(self.dna_segment_size - solution[p1][3])
                        indel_src4 = np.abs(self.dna_segment_size - solution[p2][3])
                        # both right
                        tmp_l1, tmp_l2 = solution[i][0], solution[i+1][0]+1
                        tmp_r1, tmp_r2 = solution[i][1]+1, solution[i+1][1]
                        if p1 < p2:
                            tmp_l3, tmp_l4 = solution[p1][0], solution[p2][0]+1
                            tmp_r3, tmp_r4 = solution[p1][1]+1, solution[p2][1]
                        else:
                            tmp_l3, tmp_l4 = solution[p1][0]+1, solution[p2][0]
                            tmp_r3, tmp_r4 = solution[p1][1], solution[p2][1]+1
                        dna_f1 = d_seq[tmp_l1:tmp_r1] if solution[i][7] == 'f' else reverse_complemnet(d_seq[tmp_l1:tmp_r1])
                        dna_f2 = d_seq[tmp_l2:tmp_r2] if solution[i+1][7] == 'f' else reverse_complemnet(d_seq[tmp_l2:tmp_r2])
                        dna_f3 = d_seq[tmp_l3:tmp_r3] if solution[p1][7] == 'f' else reverse_complemnet(d_seq[tmp_l3:tmp_r3])
                        dna_f4 = d_seq[tmp_l4:tmp_r4] if solution[p2][7] == 'f' else reverse_complemnet(d_seq[tmp_l4:tmp_r4])
                        indel_tgt1 = np.abs(self.dna_segment_size - len(dna_f1))
                        indel_tgt2 = np.abs(self.dna_segment_size - len(dna_f2))
                        indel_tgt3 = np.abs(self.dna_segment_size - len(dna_f3))
                        indel_tgt4 = np.abs(self.dna_segment_size - len(dna_f4))
                        d1 = Lev.distance(dna_f1, dna_f3)
                        d2 = Lev.distance(dna_f2, dna_f4)
                        if d1 + d2 < solution[i][-2] + solution[i+1][-2] or (d1+d2 == solution[i][-2] + solution[i+1][-2] and indel_src1+indel_src2+indel_src3+indel_src4 > indel_tgt1+indel_tgt2+indel_tgt3+indel_tgt4):
                            solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[i+1][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[i+1][-2] = d1, d2
                            solution[i][2], solution[i+1][2] = dna_f1, dna_f2
                            
                            solution[p1][0], solution[p2][0] = tmp_l3, tmp_l4
                            solution[p1][1], solution[p2][1] = tmp_r3, tmp_r4
                            solution[p1][3] = tmp_r3 - tmp_l3
                            solution[p2][3] = tmp_r4 - tmp_l4
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[p1][2], solution[p2][2] = dna_f3, dna_f4
                            changed = True
                            break
                        # left right
                        tmp_l1, tmp_l2 = solution[i][0], solution[i+1][0]-1
                        tmp_r1, tmp_r2 = solution[i][1]-1, solution[i+1][1]
                        if p1 < p2:
                            tmp_l3, tmp_l4 = solution[p1][0], solution[p2][0]+1
                            tmp_r3, tmp_r4 = solution[p1][1]+1, solution[p2][1]
                        else:
                            tmp_l3, tmp_l4 = solution[p1][0]+1, solution[p2][0]
                            tmp_r3, tmp_r4 = solution[p1][1], solution[p2][1]+1
                        dna_f1 = d_seq[tmp_l1:tmp_r1] if solution[i][7] == 'f' else reverse_complemnet(d_seq[tmp_l1:tmp_r1])
                        dna_f2 = d_seq[tmp_l2:tmp_r2] if solution[i+1][7] == 'f' else reverse_complemnet(d_seq[tmp_l2:tmp_r2])
                        dna_f3 = d_seq[tmp_l3:tmp_r3] if solution[p1][7] == 'f' else reverse_complemnet(d_seq[tmp_l3:tmp_r3])
                        dna_f4 = d_seq[tmp_l4:tmp_r4] if solution[p2][7] == 'f' else reverse_complemnet(d_seq[tmp_l4:tmp_r4])
                        indel_tgt1 = np.abs(self.dna_segment_size - len(dna_f1))
                        indel_tgt2 = np.abs(self.dna_segment_size - len(dna_f2))
                        indel_tgt3 = np.abs(self.dna_segment_size - len(dna_f3))
                        indel_tgt4 = np.abs(self.dna_segment_size - len(dna_f4))
                        d1 = Lev.distance(dna_f1, dna_f3)
                        d2 = Lev.distance(dna_f2, dna_f4)
                        if d1 + d2 < solution[i][-2] + solution[i+1][-2] or (d1+d2 == solution[i][-2] + solution[i+1][-2] and indel_src1+indel_src2+indel_src3+indel_src4 > indel_tgt1+indel_tgt2+indel_tgt3+indel_tgt4):
                            solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[i+1][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[i+1][-2] = d1, d2
                            solution[i][2], solution[i+1][2] = dna_f1, dna_f2
                            
                            solution[p1][0], solution[p2][0] = tmp_l3, tmp_l4
                            solution[p1][1], solution[p2][1] = tmp_r3, tmp_r4
                            solution[p1][3] = tmp_r3 - tmp_l3
                            solution[p2][3] = tmp_r4 - tmp_l4
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[p1][2], solution[p2][2] = dna_f3, dna_f4
                            changed = True
                            break
                        # left left
                        tmp_l1, tmp_l2 = solution[i][0], solution[i+1][0]-1
                        tmp_r1, tmp_r2 = solution[i][1]-1, solution[i+1][1]
                        if p1 < p2:
                            tmp_l3, tmp_l4 = solution[p1][0], solution[p2][0]-1
                            tmp_r3, tmp_r4 = solution[p1][1]-1, solution[p2][1]
                        else:
                            tmp_l3, tmp_l4 = solution[p1][0]-1, solution[p2][0]
                            tmp_r3, tmp_r4 = solution[p1][1], solution[p2][1]-1
                        dna_f1 = d_seq[tmp_l1:tmp_r1] if solution[i][7] == 'f' else reverse_complemnet(d_seq[tmp_l1:tmp_r1])
                        dna_f2 = d_seq[tmp_l2:tmp_r2] if solution[i+1][7] == 'f' else reverse_complemnet(d_seq[tmp_l2:tmp_r2])
                        dna_f3 = d_seq[tmp_l3:tmp_r3] if solution[p1][7] == 'f' else reverse_complemnet(d_seq[tmp_l3:tmp_r3])
                        dna_f4 = d_seq[tmp_l4:tmp_r4] if solution[p2][7] == 'f' else reverse_complemnet(d_seq[tmp_l4:tmp_r4])
                        indel_tgt1 = np.abs(self.dna_segment_size - len(dna_f1))
                        indel_tgt2 = np.abs(self.dna_segment_size - len(dna_f2))
                        indel_tgt3 = np.abs(self.dna_segment_size - len(dna_f3))
                        indel_tgt4 = np.abs(self.dna_segment_size - len(dna_f4))
                        d1 = Lev.distance(dna_f1, dna_f3)
                        d2 = Lev.distance(dna_f2, dna_f4)
                        if d1 + d2 < solution[i][-2] + solution[i+1][-2] or (d1+d2 == solution[i][-2] + solution[i+1][-2] and indel_src1+indel_src2+indel_src3+indel_src4 > indel_tgt1+indel_tgt2+indel_tgt3+indel_tgt4):
                            solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[i+1][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[i+1][-2] = d1, d2
                            solution[i][2], solution[i+1][2] = dna_f1, dna_f2
                            
                            solution[p1][0], solution[p2][0] = tmp_l3, tmp_l4
                            solution[p1][1], solution[p2][1] = tmp_r3, tmp_r4
                            solution[p1][3] = tmp_r3 - tmp_l3
                            solution[p2][3] = tmp_r4 - tmp_l4
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[p1][2], solution[p2][2] = dna_f3, dna_f4
                            changed = True
                            break
                        # right left
                        tmp_l1, tmp_l2 = solution[i][0], solution[i+1][0]+1
                        tmp_r1, tmp_r2 = solution[i][1]+1, solution[i+1][1]
                        if p1 < p2:
                            tmp_l3, tmp_l4 = solution[p1][0], solution[p2][0]-1
                            tmp_r3, tmp_r4 = solution[p1][1]-1, solution[p2][1]
                        else:
                            tmp_l3, tmp_l4 = solution[p1][0]-1, solution[p2][0]
                            tmp_r3, tmp_r4 = solution[p1][1], solution[p2][1]-1
                        dna_f1 = d_seq[tmp_l1:tmp_r1] if solution[i][7] == 'f' else reverse_complemnet(d_seq[tmp_l1:tmp_r1])
                        dna_f2 = d_seq[tmp_l2:tmp_r2] if solution[i+1][7] == 'f' else reverse_complemnet(d_seq[tmp_l2:tmp_r2])
                        dna_f3 = d_seq[tmp_l3:tmp_r3] if solution[p1][7] == 'f' else reverse_complemnet(d_seq[tmp_l3:tmp_r3])
                        dna_f4 = d_seq[tmp_l4:tmp_r4] if solution[p2][7] == 'f' else reverse_complemnet(d_seq[tmp_l4:tmp_r4])
                        indel_tgt1 = np.abs(self.dna_segment_size - len(dna_f1))
                        indel_tgt2 = np.abs(self.dna_segment_size - len(dna_f2))
                        indel_tgt3 = np.abs(self.dna_segment_size - len(dna_f3))
                        indel_tgt4 = np.abs(self.dna_segment_size - len(dna_f4))
                        d1 = Lev.distance(dna_f1, dna_f3)
                        d2 = Lev.distance(dna_f2, dna_f4)
                        if d1 + d2 < solution[i][-2] + solution[i+1][-2] or (d1+d2 == solution[i][-2] + solution[i+1][-2] and indel_src1+indel_src2+indel_src3+indel_src4 > indel_tgt1+indel_tgt2+indel_tgt3+indel_tgt4):
                            solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[i+1][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[i+1][-2] = d1, d2
                            solution[i][2], solution[i+1][2] = dna_f1, dna_f2
                            
                            solution[p1][0], solution[p2][0] = tmp_l3, tmp_l4
                            solution[p1][1], solution[p2][1] = tmp_r3, tmp_r4
                            solution[p1][3] = tmp_r3 - tmp_l3
                            solution[p2][3] = tmp_r4 - tmp_l4
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[p1][2], solution[p2][2] = dna_f3, dna_f4
                            changed = True
                            break
                    
                    # check if a better match when move edge to left side
                    tmp_l1, tmp_l2 = solution[i][0], solution[i+1][0]-1
                    tmp_r1, tmp_r2 = solution[i][1]-1, solution[i+1][1]                    
                    dnaf1 = d_seq[tmp_l1:tmp_r1]
                    dnaf2 = d_seq[tmp_l2:tmp_r2]
                    if solution[i][7] == 'r':
                        dnaf1 = reverse_complemnet(dnaf1)
                    if solution[i+1][7] == 'r':
                        dnaf2 = reverse_complemnet(dnaf2)
                    d1 = Lev.distance(dnaf1, solution[p1][2])
                    d2 = Lev.distance(dnaf2, solution[p2][2])
                    if d1 + d2 < solution[i][-2] + solution[i+1][-2]:
                        solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                        solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                        solution[i][3] = tmp_r1 - tmp_l1
                        solution[i+1][3] = tmp_r2 - tmp_l2
                        solution[i][-2], solution[i+1][-2] = d1, d2
                        solution[p1][-2], solution[p2][-2] = d1, d2
                        solution[i][2], solution[i+1][2] = dnaf1, dnaf2
                        changed = True
                        break
                    elif d1 + d2 == solution[i][-2] + solution[i+1][-2]:
                        # check if indel num drops
                        indel_tgt1 = np.abs(self.dna_segment_size - (tmp_r1 - tmp_l1))
                        indel_tgt2 = np.abs(self.dna_segment_size - (tmp_r2 - tmp_l2))
                        indel_src1 = np.abs(self.dna_segment_size - solution[i][3])
                        indel_src2 = np.abs(self.dna_segment_size - solution[i+1][3])

                        if indel_tgt1 + indel_tgt2 < indel_src1 + indel_src2:
                            solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[i+1][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[i+1][-2] = d1, d2
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[i][2], solution[i+1][2] = dnaf1, dnaf2
                            changed = True
                            break
                        
                    # check if a better match when move edge to right side
                    tmp_l1, tmp_l2 = solution[i][0], solution[i+1][0]+1
                    tmp_r1, tmp_r2 = solution[i][1]+1, solution[i+1][1]
                    dnaf1 = d_seq[tmp_l1:tmp_r1]
                    dnaf2 = d_seq[tmp_l2:tmp_r2]
                    if solution[i][7] == 'r':
                        dnaf1 = reverse_complemnet(dnaf1)
                    if solution[i+1][7] == 'r':
                        dnaf2 = reverse_complemnet(dnaf2)
                    d1 = Lev.distance(dnaf1, solution[p1][2])
                    d2 = Lev.distance(dnaf2, solution[p2][2])
                    if d1 + d2 < solution[i][-2] + solution[i+1][-2]:
                        solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                        solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                        solution[i][-2], solution[i+1][-2] = d1, d2
                        solution[p1][-2], solution[p2][-2] = d1, d2
                        solution[i][3] = tmp_r1 - tmp_l1
                        solution[i+1][3] = tmp_r2 - tmp_l2
                        solution[i][2], solution[i+1][2] = dnaf1, dnaf2
                        changed = True
                        break
                    elif d1 + d2 == solution[i][-2] + solution[i+1][-2]:
                        indel_tgt1 = np.abs(self.dna_segment_size - (tmp_r1 - tmp_l1))
                        indel_tgt2 = np.abs(self.dna_segment_size - (tmp_r2 - tmp_l2))
                        indel_src1 = np.abs(self.dna_segment_size - solution[i][3])
                        indel_src2 = np.abs(self.dna_segment_size - solution[i+1][3])
                        if indel_tgt1 + indel_tgt2 < indel_src1 + indel_src2:
                            solution[i][0], solution[i+1][0] = tmp_l1, tmp_l2
                            solution[i][1], solution[i+1][1] = tmp_r1, tmp_r2
                            solution[i][3] = tmp_r1 - tmp_l1
                            solution[i+1][3] = tmp_r2 - tmp_l2
                            solution[i][-2], solution[i+1][-2] = d1, d2
                            solution[p1][-2], solution[p2][-2] = d1, d2
                            solution[i][2], solution[i+1][2] = dnaf1, dnaf2
                            changed = True
                            break
                    
                
            if not changed:
                break
        coverage = np.sum([f[3] for f in solution])
        return solution, np.sum([s[-2] for s in solution])/2 + len(d_seq) - coverage
        
    def fragment_mapping(self, fixed_map, d_seq, debug=False, time_stamp=None):

        fragment_forward = [f[2] for f in fixed_map if f[-2] == 'f']
        fragment_forward_num = len(fragment_forward)
        fragment_backward = [f[2] for f in fixed_map if f[-2] == 'r']
        fragment_backward_num = len(fragment_backward)
        if fragment_forward_num != fragment_backward_num:
            sys.stderr.write('fragment number not equal\n')
            # sys.stderr.write('fragment_forward_num: {}\n'.format(fragment_forward_num))
            # sys.stderr.write('fragment_backward_num: {}\n'.format(fragment_backward_num))
            # sys.stderr.write('fixed_map:\n')
            # for f in fixed_map:
            #     sys.stderr.write(f'{f}\n')
            return []
    
        distance_matrix = np.zeros((len(fragment_forward), len(fragment_backward)))
        for i in range(len(fragment_forward)):
            for j in range(len(fragment_backward)):
                distance_matrix[i,j] = Lev.distance(fragment_forward[i], fragment_backward[j])
        
        mapping1 = []
        for i in range(distance_matrix.shape[0]):
            tmp_options = []
            for j in range(distance_matrix.shape[1]):
                if distance_matrix[i,j] == np.min(distance_matrix[:, j]):
                    tmp_options.append((i, j))
            options = []
            if len(tmp_options) > 0:
                t = np.min([np.max([distance_matrix[x, y] for x, y in tmp_options]), np.min([distance_matrix[x, y] for x, y in tmp_options])+2])
                # t = np.max([distance_matrix[x, y] for x, y in tmp_options])
                for j in range(distance_matrix.shape[1]):
                    if distance_matrix[i,j] <= t:
                        options.append((i, j))
            elif len(tmp_options) == 0:
                sorted_points = sorted(list(set(distance_matrix[i,:])))
                limit = sorted_points[:2][-1]
                for j in range(distance_matrix.shape[1]):
                    if distance_matrix[i,j] <= limit:
                        options.append((i, j))
            # print('i', i, tmp_options, options)
            mapping1 += options
        mapping2 = []
        for j in range(distance_matrix.shape[1]):
            tmp_options = []
            for i in range(distance_matrix.shape[0]):
                if distance_matrix[i,j] == np.min(distance_matrix[i, :]):
                    tmp_options.append((i, j))
            options = []
            if len(tmp_options) > 0:
                # t = np.max([distance_matrix[x, y] for x, y in tmp_options])
                t = np.min([np.max([distance_matrix[x, y] for x, y in tmp_options]), np.min([distance_matrix[x, y] for x, y in tmp_options])+2])
                for i in range(distance_matrix.shape[0]):
                    if distance_matrix[i,j] <= t:
                        options.append((i, j))
            elif len(tmp_options) == 0:
                sorted_points = sorted(list(set(distance_matrix[:,j])))
                limit = sorted_points[:2][-1]
                for i in range(distance_matrix.shape[0]):
                    if distance_matrix[i,j] <= limit:
                        options.append((i, j))
            # print('j', j, tmp_options, options)
            mapping2 += options

        mapping_list = set(mapping1).union(set(mapping2))
        mapping_list = sorted(mapping_list, key=lambda x: x[0])
        mapping = []
        xg = 0
        options = []
        for x,y in mapping_list:
            if x == xg:
                options.append((x,y))
            else:
                mapping.append(options)
                options = [(x,y)]
                xg = x
        mapping.append(options)
        if debug:
            print('fragment mapping'.center(80, '='))
            print(distance_matrix)
            for opt in mapping:
                print(opt)
                
        solutions = [[]]
        for i in range(len(fragment_forward)):
            if self.timeout is not None:
                if time.time() - time_stamp > self.timeout:
                    return []
            tmp_solutions = []
            for solution in solutions:
                options = mapping[i]
                for fidx, bidx in options:
                    b_idx = fragment_forward_num + bidx
                    if b_idx not in [frag[-1] for frag in solution]:
                        new_solution = copy.deepcopy(solution)
                        new_solution.append(copy.deepcopy(fixed_map[fidx]) + [distance_matrix[fidx][bidx], b_idx])
                        new_solution.append(copy.deepcopy(fixed_map[b_idx]) + [distance_matrix[fidx][bidx], fidx])
                        tmp_solutions.append(new_solution)
            solutions = tmp_solutions

        solutions = [sorted(solution, key=lambda x:x[0]) for solution in solutions]
        outstanding_solution = []
        min_mut_num = 1e3
        for solution in solutions:
            rc_seg_order = [solution[self.fragment_num+idx][-1] for idx in range(self.fragment_num)]
            mut_num = np.sum([seq[-2] for seq in solution]) / 2
            midx = factorial_num_to_decimal(order_seq_to_ftnum(rc_seg_order))

            # if mut_num < self.dna_segment_size * self.fragment_num * 0.3:
            outstanding_solution.append([solution, midx, mut_num])

        if debug:
            print('Raw mapping'.center(80, '='))
            for solution, midx, mnum in outstanding_solution:
                print(f'split line {midx, mnum}'.center(90, '-'))
                for seq in solution:
                    print(seq)
                    
        if len(outstanding_solution) == 0:
            return []

        rt_solutions = []
        for solution, midx, mut_num in outstanding_solution:
            if self.timeout is not None:
                if time.time() - time_stamp > self.timeout:
                    break
            updated_solution, updated_mut_num = self.polish_solution(solution, d_seq)
            if updated_mut_num < self.dna_segment_size * self.fragment_num * 0.20 and (updated_solution, midx, updated_mut_num) not in rt_solutions:
                rt_solutions.append((updated_solution, midx, updated_mut_num))
        if debug:
            print(f'final mapping solutions {len(rt_solutions)}'.center(90, '-'))
            for solution, midx, mut_num in rt_solutions:
                print(f'solution {midx, mut_num}'.center(90, '-'))
                for seq in solution:
                    print(seq)
        return rt_solutions


    def get_new_segments(self, dseg1, dseg2):
        if '' == dseg1:
            return [(dseg2, 0)]
        elif '' == dseg2:
            return [(dseg1, 0)]
            
        alns = sorted(self.aligner(dseg1, dseg2), key=lambda x: x.score, reverse=True)
        max_score = alns[0].score
    
        # get options for each site
        new_segments = []
        delta_err_nums = []
        for aln in alns:
            if max_score > aln.score:
                continue
            bases = []
            query, target = aln.seqA, aln.seqB
            aligned_str_len = len(query)
            match_num = 0
            for idx in range(len(query)):
                if query[idx] == target[idx]:
                    bases.append((query[idx], 0))
                    match_num += 1
                elif '-' not in [query[idx], target[idx]]:
                    bases.append([(query[idx], 0), (target[idx], 0), (target[idx]+query[idx], 1), (query[idx]+target[idx], 1), ('', 1)])
                else:
                    bases.append([(query[idx], 0), (target[idx], 0)])
            if match_num == aligned_str_len:
                delta_err_num = 2 * np.abs(self.dna_segment_size - match_num)
                delta_err_nums.append(delta_err_num)
            else:
                delta_err_num = 0
            # get all combinations
            segments = [['', delta_err_num]]
            for bs in bases:
                if isinstance(bs, list):
                    tmp = []
                    for b in bs:
                        tmp1 = copy.deepcopy(segments)
                        for sidx in range(len(tmp1)):
                            tmp1[sidx][0] += b[0]
                            tmp1[sidx][1] += b[1]
                        tmp += tmp1
                    segments = tmp
                else:
                    for sidx in range(len(segments)):
                        segments[sidx][0] += bs[0]
                        segments[sidx][1] += bs[1]
            
            # remove gaps
            new_segments += [(re.sub('-', '', seg[0]), seg[1]+delta_err_num) for seg in segments]
        
        new_segments = [item for item in new_segments if len(item[0]) == self.dna_segment_size]
        if len(new_segments) > 0:
            return sorted(list(set(new_segments)), key=lambda x: x[1])
        else:
            min_dis = 0 if len(delta_err_nums) == 0 else np.min(delta_err_nums)
            return [(dseg1, min_dis)]
    
    def dispose_seg_pair(self, dseg1, dseg2):
        # get dna segments
        dna_segment_1 = dseg1
        dna_segment_2 = dseg2
    
        new_segments = self.get_new_segments(dna_segment_1, dna_segment_2)
        
        # traverse new segments
        candidates = {
            10: [],
            1: [],
            -1: []
        }
        for newseg, delta_m_num in new_segments:
            status, ci = self.get_decode_status(newseg)
            # if status in [11, 10]:
            candidates[status].append((status, ci, newseg, delta_m_num))
    
        if len(candidates[10]) > 0:
            return [(item[1], item[2], item[0], item[3]) for item in candidates[10]]
        elif len(candidates[1]) > 0:
            return [(item[1], item[2], item[0], item[3]) for item in candidates[1]]
        else:
            return [(get_random_binary_seq(self.ebi_segment_size), item[2], item[0], item[3]) for item in candidates[-1]]
            # return [(get_random_binary_seq(self.ebi_segment_size), dna_segment_1, -1, Lev.distance(dna_segment_1, dna_segment_2))]
            
    def extract_binary(self, mutated_dna, debug=False, check_num=2):
    
        time_stamp = time.time()
        
        fragments, merged_fragments, maps = self.getMaps(mutated_dna, time_stamp)
        
        if self.timeout is not None:
            time_span = time.time() - time_stamp
            if time_span > self.timeout:
                # sys.stderr.write(f'Timeout \n')
                return []
    
        if debug:
            print(f'Fragments'.center(80, '='))
            for f in fragments:
                print(f)

            print(f'merged_fragments'.center(80, '='))
            for mf in merged_fragments:
                print(mf)
            
        extracted_info = []
        polished_solutions = []
        final = []
        for fmap in maps:
                
            if self.timeout is not None:
                time_span = time.time() - time_stamp
                if time_span > self.timeout:
                    # sys.stderr.write(f'Timeout \n')
                    return sorted(final, key=lambda x: x[-2])
                    
            if debug:
                print(f'Raw map'.center(80, '-'))
                for f in fmap:
                    print(f)
            
            psolutions = self.fragment_mapping(fmap, mutated_dna, debug, time_stamp)
            # random.shuffle(psolutions)
            
            for item in psolutions:
                if item not in polished_solutions:
                    polished_solutions.append(item)
                    
        polished_solutions = sorted(polished_solutions, key=lambda x: x[-1])
        
        for solution, midx, mnum in polished_solutions[:10]:
            if self.timeout is not None:
                time_span = time.time() - time_stamp
                if time_span > self.timeout:
                    # sys.stderr.write(f'Timeout \n')
                    return sorted(final, key=lambda x: x[-2])
                    
            if debug:
                print(f'polished solution {midx, mnum}'.center(80, '-'))
                for f in solution:
                    print(f)
            
            # solution, mnum = validation(solution, mutated_dna, fragments, fragment_size, fragment_num, check_threshold, debug)
    
            options = []
            coverage = np.sum([s[3] for s in solution])
            for s in solution[:self.fragment_num]:
                p = s[-1]
                seg1, seg2 = s[2], solution[p][2]
                rz = self.dispose_seg_pair(seg1, seg2)
                options.append(rz)
    
            if debug:
                print(f'binary information options'.center(80, '-'))
                for opts in options:
                    print(opts)
            
            binfos = [['', [], 0]]
            for fidx, option in enumerate(options):
                tmp_info = []
                for opt in option:
                    tmp_binfo = copy.deepcopy(binfos)
                    for binfo in tmp_binfo:
                        binfo[0] += opt[0]
                        if opt[2] < 10: # or len(option) > 1:
                            binfo[1].append(fidx)
                        binfo[2] += opt[3]
                    tmp_info += tmp_binfo
                binfos = tmp_info
    
            for bif in binfos:
                bis, error_pos, d_err = bif
                if (bis, midx, coverage, mnum, d_err) not in extracted_info:
                    extracted_info.append((bis, midx, coverage, mnum, d_err))
                    
                    if debug:
                        print(f'raw binary: {bis}')
                    
                    if self.encryption:
                        random_info = ''.join(self.get_randomfactors(midx))
                        bis = XOR2([bis, random_info])
        
                    if debug:
                        print(f'decryption: {bis}')
                    
                    if self.rs is not None:
                        if self.rs_item_num >= len(error_pos) > 0:
                            bis = self.remove_rs_info(bis, error_pos)
                        else:
                            bis = self.remove_rs_info(bis)
        
                    if debug:
                        print(f'RSdecoding: {bis}')
                    
                    if not bis:
                        continue

                    # double check 
                    new_dna = self.get_DNA_molecule(bis, midx)
                    g_mut_num = Lev.distance(new_dna, mutated_dna)
                    mut_total = mnum + d_err
        
                    if debug:
                        print(f'double check: {g_mut_num} {mut_total}')
                    
                    if  abs(Lev.distance(new_dna, mutated_dna)-mut_total) < 3:
                        rz = (bis, midx, coverage, g_mut_num, mut_total)
                        if rz not in final:
                            final.append(rz)
                        if check_num is not None and check_num <= len(final):
                            return sorted(final, key=lambda x: x[-2])
                                
        return sorted(final, key=lambda x: x[-2])
        
    def decode(self, DNA_set, output_file, file_key=1812433257, file_bit_size=None, seg_dict=None, debug=False):
        """
        Transform a set of DNA molecules to a binary file
        
        Parameters:
        DNA_set (list[str]): a list of DNA molecules.
        output_file (str): the path to output file
        file_key (int): the key for dencryption.
        file_bit_size (int): bit number of the source file
        seg_dict (dict{idx:[(binary_info, count)]}): decoding dict, use for multi-step decoding or parallel decoding
        debug (bool): print the debug information to stdout
        
        Returns:
        dict{idx:[[binary_info, count]]}: updated seg_dict
        list[str] : decision of decoding, voting result of the updated seg_dict
        """
        self.rng.update(prime=file_key)
        
        fragment_xor_num  = self.batch_size
        if file_bit_size is None:
            expected_sequence_number = None
        else:
            expected_sequence_number = int(np.ceil(file_bit_size / self.pure_ebi_size / self.batch_size)) * (self.batch_size+1)
        
        if seg_dict is None:
            seg_dict = {}
            
        progress = 0
        sys.stderr.write(f'Extract infomation from DNA molecules ...\n')
        
        queue = multiprocessing.Queue()
        params = [{'mutated_dna':dna, 'check_num':1} for dna in DNA_set]
        num_workers = 16
        task_id = 0
        drop_num = 0
        while 1:
            processes = []  # Reset processes list for each batch
    
            # Start worker processes up to num_workers
            for _ in range(num_workers):
                if task_id < len(params):
                    p = multiprocessing.Process(target=worker, args=(self.extract_binary, queue), kwargs=params[task_id])
                    processes.append(p)
                    p.start()
                    task_id += 1
                else:
                    break  # Exit if no more tasks
    
            # Wait for processes to finish and manage their status
            for p in processes:
                p.join(self.timeout)
                if p.is_alive():  # Check if the process is still alive
                    p.terminate()  # Terminate the process if it's still running
                    p.join()  # Ensure the process has been cleaned up
                else:
                    rz = queue.get()  # Retrieve the result if completed
                    if len(rz) == 0:
                        drop_num += 1
                        continue
                    if len(rz) == 1:
                        bis, seg_idx, coverage, mnum, _, mutated_dna = rz[0]
                        # bis, seg_idx, d1, d2 = rz[0]
                        dna = self.get_DNA_molecule(bis, seg_idx)
                        if dna in seg_dict:
                            uniq = True
                            for info_idx in range(len(seg_dict[dna])):
                                bs = seg_dict[dna][info_idx][0]
                                if bs == bis:
                                    seg_dict[dna][info_idx][2] += 1
                                    seg_dict[dna][info_idx][3].append((mnum, mutated_dna))
                                    uniq = False
                                    break
                            if uniq:
                                seg_dict[dna].append([bis, seg_idx, 1, [(mnum, mutated_dna)]])
                        else:
                            if expected_sequence_number is not None:
                                if seg_idx < expected_sequence_number:
                                    seg_dict[dna] = [[bis, seg_idx, 1, [(mnum, mutated_dna)]]]
                                else:
                                    if debug:
                                        sys.stderr.write(f'Molecule {idx} has a ivalid molecule index!\n')
                            else:
                                seg_dict[dna] = [[bis, seg_idx, 1, [(mnum, mutated_dna)]]]
                    else:
                        for bis, seg_idx, coverage, mnum, _, mutated_dna in rz:
                            dna = self.get_DNA_molecule(bis, seg_idx)
                            if dna in seg_dict:
                                uniq = True
                                for info_idx in range(len(seg_dict[dna])):
                                    bs = seg_dict[dna][info_idx][0]
                                    if bs == bis:
                                        seg_dict[dna][info_idx][2] += 1
                                        seg_dict[dna][info_idx][3].append((mnum, mutated_dna))
                                        uniq = False
                                        break
                                if uniq:
                                    seg_dict[dna].append([bis, seg_idx, 1, [(mnum, mutated_dna)]])
                            else:
                                if expected_sequence_number is not None:
                                    if seg_idx < expected_sequence_number:
                                        seg_dict[dna] = [[bis, seg_idx, 1, [(mnum, mutated_dna)]]]
                                    else:
                                        if debug:
                                            sys.stderr.write(f'Molecule {idx} has a ivalid molecule index!\n')
                                else:
                                    seg_dict[dna] = [[bis, seg_idx, 1, [(mnum, mutated_dna)]]]
                    
            # progress control
            if expected_sequence_number is not None:
                progress = len(list(seg_dict.keys())) / expected_sequence_number * 100
                sys.stderr.write(f'\rProgress: {progress:.2f}% {len(seg_dict)} / {expected_sequence_number} ... on {task_id} / {len(DNA_set)} drop num: {drop_num}')
                if progress >= 100:
                    break
            else:
                progress = (task_id)/len(DNA_set)*100
                sys.stderr.write(f'\rProgress: {progress:.2f}% {task_id} / {len(DNA_set)} ... drop num: {drop_num}')
    
            # Break the loop if all tasks have been processed
            if task_id >= len(params) or progress >= 100:
                break

        # sys.stderr.write(f'\rProgress: {progress:.2f} %\n')
        sys.stderr.write(f'Write information to output file ...\n')

        # output result to output file
        binary_seq = []
        complete = True
        selected_binary = {}
        new_seg_dict = {}
        for kw in seg_dict:
            sorted_list = sorted(seg_dict[kw], key=lambda x: -x[2])
            bis = sorted_list[0][0]
            idx = sorted_list[0][1]
            selected_binary[idx] = bis
        
        if expected_sequence_number is not None:
            maxi_seg_index = expected_sequence_number
        else:
            maxi_seg_index = int(np.max(list(selected_binary.keys())))+1
        for idx in range(0, maxi_seg_index, fragment_xor_num+1):
            for shift in range(0, fragment_xor_num):
                if idx+shift not in selected_binary:
                    try:
                        batch = set([idx+st for st in range(fragment_xor_num + 1)]) - set([idx+shift])
                        binfos = [selected_binary[fidx] for fidx in list(batch)]
                        selected_binary[idx+shift] = XOR2(binfos)

                    except:
                        sys.stderr.write(f'molecule {idx+shift} is missed. And the batch is not complete.\n')
                        complete = False
                    else:
                        binary_seq += selected_binary[idx+shift]
                else:
                    binary_seq += selected_binary[idx+shift]

        if complete:
            if file_bit_size is None:
                save_binary_info(binary_seq, None, output_file)
            else:
                save_binary_info(binary_seq, int(file_bit_size/8), output_file)
        else:
            sys.stderr.write(f'The file is not complete. Need to collect more data.\n')
        sys.stderr.write(f'Done!\n')
        
        return seg_dict, selected_binary
