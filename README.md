# *Brian 2* benchmarks

This repository hosts runtime and memory benchmarks for the
[Brian 2 simulator](https://github.com/brian-team/brian2). The results are
available here: https://brian-team.github.io/brian2_benchmarks/

Benchmarking is done with the tool
[*airspeed velocity* (`asv`)](http://asv.readthedocs.io).
 
## Writing benchmarks
Refer to the [airspeed velocity documentation](http://asv.readthedocs.io/en/latest/writing_benchmarks.html)
for general explanations about writing benchmarks. A few comments specific to
the Brian 2 simulator:
* Brian has a system that caches intermediate results for code generation. The
  `clear_cache` function in `benchmarks/benchmarks.py` will clear this cache
  to avoid that it interferes with the measurements. However, the `setup`
  function responsible for doing this kind of preparation work is only called
  for each "repeat", not for the "number" of runs (see Python's
  [timeit documentation](https://docs.python.org/3/library/timeit.html) for
  details). We therefore explicitly set `number` to 1.
* Similarly, we delete the on-disk caches for runtime targets, and generate
  standalone code in fresh temporary directories each time.
* For tests run with the standalone target, we have to set the `timer`
  explicitly (e.g. to `timeit.default_timer`), because the default timer used by
  `asv` only measures the time in the main process. Since the actual standalone
  simulation is run in a subprocess, the time spent during the actual simulation
  would not be counted!  


## Running benchmarks
Again, the [airspeed velocity documentation](http://asv.readthedocs.io/en/latest/using.html#running-benchmarks)
has most of the necessary information. A useful command during development of
new benchmarks is to run benchmarks only for the latest revision, the current
Python version and with a single repetition:
```console
$ asv dev
```

Note that `asv run` takes arguments in the same way as `git log`, i.e. it will
by default run benchmarks for a *range* of revisions. To run benchmarks for a
single revision, you can use the `^!` shorthand, e.g. to run benchmarks only
for the `2.1.2` release:
```console
$ asv run 2.1.2^!
```