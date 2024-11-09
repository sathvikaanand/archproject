#
# Copyright 2017 Andrey Rodchenko, School of Computer Science, The University of Manchester
# Parts of this script were obtained from the ZSim-NVMain simulator 
# (https://github.com/AXLEproject/axle-zsim-nvmain/blob/master/misc/zsim_lib.py).
# 
# This code is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 only, as
# published by the Free Software Foundation.
#                                                                          
# This code is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# version 2 for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np
import re

def convert_config_to_json(js):
  js = '{\n%s}' % js
  js = re.sub(' = ', ' : ', js)
  js = re.sub(';\n', ',\n', js)
  js = re.sub(r',\n(\s*)}', r'\n\1}', js)
  js = re.sub(r'(\S+) : ', r'"\1" : ', js)
  js = re.sub(r'(\d+)L', r'\1', js)
  return js

def sys_fs_to_cycles(a):
  return float(a['sys']['frequency']) / 1e9

def total_cores(a):
  out = 0
  for core in a['sys']['cores'].values():
    out += int(core['cores'])
  return str(out)

def cache_levels(a):
  last_level = 0
  for k, v in a['sys']['caches'].items():
    assert(k.lower().startswith('l'))
    curr_level = int(k[1:2])
    if curr_level > last_level:
      last_level = curr_level
  return last_level

def cores_num(a):
  all_cores, = a['sys']['cores'].values()
  return int(all_cores['cores'])

def cache_shared_cores(a, cache_name):
  cache = a['sys']['caches'][cache_name]
  return str(cores_num(a) // int(cache['caches']))

def cache_dvfs_domain(a, cache_name):
  if cache_shared_cores(a, cache_name) == '1':
    return 'core'
  else:
    return 'global'

def core_type(a):
  all_cores, = a['sys']['cores'].values()
  if all_cores['type'] == 'OOO':
    return 'rob'
  else:
    return 'interval'

def rob_timer_in_order(a):
  all_cores, = a['sys']['cores'].values()
  if all_cores['type'] == 'Simple':
    return '1'
  else:
    return '0'

def cache_load_hits(a, dset, zcn):
  is_l1 = zcn in 'l1d l1i'.split()
  return ((dset[-1][zcn]['fhGETS'] if is_l1 else 0) +
      dset[-1][zcn]['hGETS'])

def cache_load_misses(a, dset, zcn):
  return dset[-1][zcn]['mGETS']

def cache_loads(a, dset, zcn):
  is_l1 = zcn in 'l1d l1i'.split()
  return ((dset[-1][zcn]['fhGETS'] if is_l1 else 0) +
      dset[-1][zcn]['hGETS'] +
      dset[-1][zcn]['mGETS'])

def cache_store_hits(a, dset, zcn):
  is_l1 = zcn in 'l1d l1i'.split()
  return ((dset[-1][zcn]['fhGETX'] if is_l1 else 0) +
      dset[-1][zcn]['hGETX'])

def cache_store_misses(a, dset, zcn):
  return (dset[-1][zcn]['mGETXIM'] + dset[-1][zcn]['mGETXSM'])

def cache_stores(a, dset, zcn):
  is_l1 = zcn in 'l1d l1i'.split()
  return ((dset[-1][zcn]['fhGETX'] if is_l1 else 0) +
      dset[-1][zcn]['hGETX'] +
      dset[-1][zcn]['mGETXIM'] +
      dset[-1][zcn]['mGETXSM'])

def core_instrs(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['instrs']

def core_cycles(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['cycles']

def cycles_num(a, dset):
  phaseLength = int(a['sim']['phaseLength'])
  phases = int(dset[-1]['phase'])
  return int(phaseLength * (phases + 1))

def elapsed_time(a, dset):
  ftc = sys_fs_to_cycles(a)
  return int(float(cycles_num(a, dset)) / ftc)

def core_idle_elapsed_time(a, dset):
  ftc = sys_fs_to_cycles(a)
  total_core_cycles = np.asarray([cycles_num(a, dset)] * cores_num(a))
  nonidle_core_cycles = core_cycles(a, dset)
  idle_core_cycles = total_core_cycles - nonidle_core_cycles
  result = np.asarray(list(map(lambda x: int(x / ftc), idle_core_cycles)))
  return result

def core_mispred_branches(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['mispredBranches']

def core_pred_branches(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['predBranches']

def core_uop_branches(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['branchUops']

def core_uop_fp_addsubs(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['fpAddSubUops']

def core_uop_fp_muldivs(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['fpMulDivUops']

def core_uop_total(a, dset):
  core_name, = a['sys']['cores'].keys()
  return dset[-1][core_name]['uops']

def core_uop_loads(a, dset):
  return cache_loads(a, dset, 'l1d')

def core_uop_stores(a, dset):
  return cache_stores(a, dset, 'l1d')

def core_uop_generics(a, dset):
  uop_non_generics = core_uop_stores(a, dset) + core_uop_loads(a, dset)
  uop_non_generics += core_uop_fp_addsubs(a, dset)
  uop_non_generics += core_uop_fp_muldivs(a, dset)
  uop_non_generics += core_uop_branches(a, dset)
  uop_total = core_uop_total(a, dset)
  uop_generics = uop_total - uop_non_generics
  return uop_generics