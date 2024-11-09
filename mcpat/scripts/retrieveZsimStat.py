#!/usr/bin/env python
# Copyright 2017 Andrey Rodchenko, School of Computer Science, The University of Manchester
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
import sys
import h5py # presents HDF5 files as numpy arrays
import os
import numpy
import json
import retrieveZSimStatLib

if len(sys.argv) < 3:
  print("The script should have at least 3 input parameters! Pass ZSim stat folder as the first parameter. Pass characterstic as the second parameter.")
  sys.exit(1)
  
zsim_stat_dir = sys.argv[1]
char = sys.argv[2]

jf = open(os.path.join(zsim_stat_dir, './.out.cfg'))
js = retrieveZSimStatLib.convert_config_to_json(jf.read()) 
a = json.loads(js)

f = h5py.File(os.path.join(zsim_stat_dir, '.zsim-ev.h5'), 'r')
dset = f['stats']['root']

if char == 'C':
  print(numpy.sum(retrieveZSimStatLib.core_cycles(a, dset)))

elif char == 'I':
  print(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset)))

elif char == 'IPC':
  print(float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))) /
        float(numpy.sum(retrieveZSimStatLib.core_cycles(a, dset))))
elif char == "CPI":
  print(float(numpy.sum(retrieveZSimStatLib.core_cycles(a, dset))) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))


elif char == 'CHLD':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(numpy.sum(retrieveZSimStatLib.cache_load_hits(a, dset, zcn)))

elif char == 'CHST':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(numpy.sum(retrieveZSimStatLib.cache_store_hits(a, dset, zcn)))

elif char == 'CHLDST':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_load_hits(a, dset, zcn)) +
        numpy.sum(retrieveZSimStatLib.cache_store_hits(a, dset, zcn))))

elif char == 'CHLDPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_load_hits(a, dset, zcn)) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))

elif char == 'CHSTPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_store_hits(a, dset, zcn)) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))

elif char == 'CHLDSTPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float((numpy.sum(retrieveZSimStatLib.cache_load_hits(a, dset, zcn)) +
        numpy.sum(retrieveZSimStatLib.cache_store_hits(a, dset, zcn))) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))

elif char == 'CMLD':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(numpy.sum(retrieveZSimStatLib.cache_load_misses(a, dset, zcn)))

elif char == 'CMST':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(numpy.sum(retrieveZSimStatLib.cache_store_misses(a, dset, zcn)))

elif char == 'CMLDST':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_load_misses(a, dset, zcn)) +
        numpy.sum(retrieveZSimStatLib.cache_store_misses(a, dset, zcn))))

elif char == 'CMLDPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_load_misses(a, dset, zcn)) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))

elif char == 'CMSTPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_store_misses(a, dset, zcn)) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))

elif char == 'CMLDSTPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float((numpy.sum(retrieveZSimStatLib.cache_load_misses(a, dset, zcn)) +
        numpy.sum(retrieveZSimStatLib.cache_store_misses(a, dset, zcn))) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))


elif char == 'CALD':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(numpy.sum(retrieveZSimStatLib.cache_loads(a, dset, zcn)))

elif char == 'CAST':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(numpy.sum(retrieveZSimStatLib.cache_stores(a, dset, zcn)))

elif char == 'CALDST':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_loads(a, dset, zcn)) +
        numpy.sum(retrieveZSimStatLib.cache_stores(a, dset, zcn))))

elif char == 'CALDPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_loads(a, dset, zcn)) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))

elif char == 'CASTPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float(numpy.sum(retrieveZSimStatLib.cache_stores(a, dset, zcn)) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))

elif char == 'CALDSTPKI':
  if len(sys.argv) < 4:
    print("The script should have the 3th cache name parameter for " + char + " characteristic.")
    sys.exit(1)
  zcn = sys.argv[3]
  print(float((numpy.sum(retrieveZSimStatLib.cache_loads(a, dset, zcn)) +
        numpy.sum(retrieveZSimStatLib.cache_stores(a, dset, zcn))) * 1000) /
        float(numpy.sum(retrieveZSimStatLib.core_instrs(a, dset))))


else:
  print(char + " characteristic is not supported!")
  print("Supported characteristics are:")
  print("  C - cycles")
  print("  I - instructions")
  print("  IPC - instructions per clock")
  print("  CPI - cycles per instruction")
  print("  C[H|M|A][LD|ST|LDST](PKI) - cache characteristics")
  print("    [..|..] - required alternatives")
  print("    (..|..) - optional alternatives")
  print("    H       - hits")
  print("    M       - misses")
  print("    A       - accesses")
  print("    LD      - loads")
  print("    ST      - stores")
  print("    LDST    - loads and stores")
  print("    PKI     - per kilo instruction")
  
  sys.exit(1)