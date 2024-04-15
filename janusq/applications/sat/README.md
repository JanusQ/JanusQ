================================================================================
Quick Install

- Notice: Please use the Linux or MacOS running these files, the Window system may 
  encounter compilation errors.
  
- Decide where to install the files . The simplest approach is to use
  GNU standard locations and just set a "prefix" for the root install
  directory (reffered to as $PREFIX below). More control can be
  achieved by overriding other of the GNU standard install locations
  (includedir, bindir, etc). Configuring with just a prefix:

  > make config prefix=$PREFIX

- Compiling and installing:

  > make install



================================================================================
Directory Overview:

minisat/mtl/            Mini Template Library
minisat/utils/          Generic helper code (I/O, Parsing, CPU-time, etc)
minisat/core/           A core version of the solver
minisat/simp/           An extended solver with simplification capabilities
minisat/tabu/           Quantum annealing code
minisat/python/         Dwave 2000Q system API for python
LICENSE

================================================================================
Examples:

Run minisat with same heuristics as version 2.0:

> minisat <cnf-root-file> -no-luby -rinc=1.5 -phase-saving=0 -rnd-freq=0.02

- <cnf-root-file> means the the directory path containing the cnf file

================================================================================
Compile Error：

- if you meet the error about "Makefile:xxx: recipe for target 'xxx' failed" in Linux, 
    you can try these steps:
> open  /etc/ld.so.conf
> add clause:  /usr/local/lib
> save file and execute command:  /sbin/ldconfig
> execute command: make install
