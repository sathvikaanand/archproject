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
import h5py # presents HDF5 files as numpy arrays
import numpy as np
import os
import json
import retrieveZSimStatLib

def convert(zsim_stat_dir, application):
  jf = open(os.path.join(zsim_stat_dir, application + '.out.cfg'))
  js = retrieveZSimStatLib.convert_config_to_json(jf.read()) 
  a = json.loads(js)

  f = h5py.File(os.path.join(zsim_stat_dir, application + '.zsim-ev.h5'), 'r')
  dset = f['stats']['root']

  def convert_cache_loads_stores(a, dset, zsim_cache_name, sniper_cache_name):
    # returns padded list
    def plist(l, n):
      return (list(l) + [0]*n)[:n]
    zcn = zsim_cache_name
    scn = sniper_cache_name
    n_cores = retrieveZSimStatLib.cores_num(a)

    load_misses = retrieveZSimStatLib.cache_load_misses(a, dset, zcn)

    loads = retrieveZSimStatLib.cache_loads(a, dset, zcn)

    store_misses = retrieveZSimStatLib.cache_store_misses(a, dset, zcn)

    stores = retrieveZSimStatLib.cache_stores(a, dset, zcn)

    out = {scn + '.load-misses': plist(load_misses, n_cores),
           scn + '.loads': plist(loads, n_cores),
           scn + '.store-misses': plist(store_misses, n_cores),
           scn + '.stores': plist(stores, n_cores)}

    if zsim_cache_name == 'l'+str(retrieveZSimStatLib.cache_levels(a)):
        out.update({'dram.reads': plist(load_misses, n_cores),
                    'dram.writes': plist(store_misses, n_cores)})

    return out
 
  results = {
    'config': {
               #FIXME: read these constants from configuration file
              'perf_model/branch_predictor/mispredict_penalty': '17',
              'perf_model/nuca/enabled': 'false',
              'perf_model/core/interval_timer/dispatch_width': '4',
              'perf_model/core/interval_timer/window_size': '128',
              'perf_model/dram/num_controllers': '1',
              'perf_model/dram/dimms_per_controller': '3',
              'perf_model/dram/chips_per_dimm': '8',
              'power/technology_node': '45',
              'power/vdd': '1.2',

              'general/total_cores': retrieveZSimStatLib.total_cores(a),
              'perf_model/cache/levels': retrieveZSimStatLib.cache_levels(a),
              'perf_model/core/frequency': str(float(a['sys']['frequency'])/1000),
              'perf_model/core/type': retrieveZSimStatLib.core_type(a),
              'perf_model/core/rob_timer/in_order': retrieveZSimStatLib.rob_timer_in_order(a),

              'perf_model/l1_dcache/associativity': a['sys']['caches']['l1d']['array']['ways'],
              'perf_model/l1_dcache/cache_block_size': a['sys']['lineSize'],
              'perf_model/l1_dcache/cache_size': str(int(a['sys']['caches']['l1d']['size'])/1024),
              'perf_model/l1_dcache/data_access_time': a['sys']['caches']['l1d']['latency'],
              'perf_model/l1_dcache/shared_cores': retrieveZSimStatLib.cache_shared_cores(a, 'l1d'),
              'perf_model/l1_dcache/dvfs_domain': retrieveZSimStatLib.cache_dvfs_domain(a, 'l1d'),

              'perf_model/l1_icache/associativity': a['sys']['caches']['l1i']['array']['ways'],
              'perf_model/l1_icache/cache_block_size': a['sys']['lineSize'],
              'perf_model/l1_icache/cache_size': str(int(a['sys']['caches']['l1i']['size'])/1024),
              'perf_model/l1_icache/data_access_time': a['sys']['caches']['l1i']['latency'],
              'perf_model/l1_icache/shared_cores': retrieveZSimStatLib.cache_shared_cores(a, 'l1i'),
              'perf_model/l1_icache/dvfs_domain': retrieveZSimStatLib.cache_dvfs_domain(a, 'l1i'),

              'perf_model/l2_cache/associativity': a['sys']['caches']['l2']['array']['ways'],
              'perf_model/l2_cache/cache_block_size': a['sys']['lineSize'],
              'perf_model/l2_cache/cache_size': str(int(a['sys']['caches']['l2']['size'])/1024),
              'perf_model/l2_cache/data_access_time': a['sys']['caches']['l2']['latency'],
              'perf_model/l2_cache/shared_cores': retrieveZSimStatLib.cache_shared_cores(a, 'l2'),
              'perf_model/l2_cache/dvfs_domain': retrieveZSimStatLib.cache_dvfs_domain(a, 'l2'),

              'perf_model/l3_cache/associativity': a['sys']['caches']['l3']['array']['ways'],
              'perf_model/l3_cache/cache_block_size': a['sys']['lineSize'],
              'perf_model/l3_cache/cache_size': str(int(a['sys']['caches']['l3']['size'])/1024),
              'perf_model/l3_cache/data_access_time': a['sys']['caches']['l3']['latency'],
              'perf_model/l3_cache/shared_cores': retrieveZSimStatLib.cache_shared_cores(a, 'l3'),
              'perf_model/l3_cache/dvfs_domain': retrieveZSimStatLib.cache_dvfs_domain(a, 'l3')
    },

   'results': {'interval_timer.uop_branch': list(retrieveZSimStatLib.core_uop_branches(a, dset)),
               'interval_timer.uop_fp_addsub': list(retrieveZSimStatLib.core_uop_fp_addsubs(a, dset)),
               'interval_timer.uop_fp_muldiv': list(retrieveZSimStatLib.core_uop_fp_muldivs(a, dset)),
               'interval_timer.uop_load': list(retrieveZSimStatLib.core_uop_loads(a, dset)),
               'interval_timer.uop_store': list(retrieveZSimStatLib.core_uop_stores(a, dset)),
               'interval_timer.uop_generic': list(retrieveZSimStatLib.core_uop_generics(a, dset)),
               'interval_timer.uops_total': list(retrieveZSimStatLib.core_uop_total(a, dset)),

               'branch_predictor.num-incorrect': list(retrieveZSimStatLib.core_mispred_branches(a, dset)),
               'branch_predictor.num-correct': list(retrieveZSimStatLib.core_pred_branches(a, dset)),

               'fs_to_cycles': str(retrieveZSimStatLib.sys_fs_to_cycles(a)),
               'fs_to_cycles_cores': [retrieveZSimStatLib.sys_fs_to_cycles(a)] * retrieveZSimStatLib.cores_num(a),
               'global.time_begin': 0,
               'global.time_end': retrieveZSimStatLib.elapsed_time(a, dset),
               'global.time': retrieveZSimStatLib.elapsed_time(a, dset),
               'performance_model.idle_elapsed_time': list(retrieveZSimStatLib.core_idle_elapsed_time(a, dset)),
               'performance_model.elapsed_time': [retrieveZSimStatLib.elapsed_time(a, dset)] * retrieveZSimStatLib.cores_num(a),
               'performance_model.instruction_count': list(retrieveZSimStatLib.core_instrs(a, dset)),

               'network.shmem-1.bus.num-packets': [0, 0],
               'network.shmem-1.bus.time-used': [0, 0],

    }
  }

  results['results'].update(convert_cache_loads_stores(a, dset, 'l1d', 'L1-D'))
  results['results'].update(convert_cache_loads_stores(a, dset, 'l1i', 'L1-I'))
  results['results'].update(convert_cache_loads_stores(a, dset, 'l2', 'L2'))
  results['results'].update(convert_cache_loads_stores(a, dset, 'l3', 'L3'))

  return results

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: %s <zsim_stat_dir> <application>" % sys.argv[0])
        sys.exit(1)
    print(json.dumps(convert(sys.argv[1], sys.argv[2]), cls=CustomEncoder, indent=4))