GPU-Parallel-FFT
================

This is a GPU-accelerated library for computing Fast Fourier Transforms
of 3-Dimensional meshes distributed among MPI processes.

Disclaimer
----------

This is an experiment, for the CSC GPU Hackathon 2020.
Use at your own risk.

Requirements
------------

- any MPI library, version?
- `cuda`, version?
- `meson` build system, version?
- `FFTW3`, only for testing correctedness.
- `Boost::test` for unit testing.


License
-------

GPL2

Authors
-------

So far just me.

Getting started
---------------

```
meson ...
```
