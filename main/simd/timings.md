P2(Tet), Laplace operator

80^3 Cube, with insert

Orig FFC:
Timings for element matrix (n=10) avg: 8854.41ms, min: 8142.04ms, max: 9153.02ms

---

80^3 Cube, no insert

Orig FFC:
Timings for element matrix (n=10) avg: 1441.0ms, min: 1424.72ms, max: 1484.37ms

All compiled with "-O2", "-march=native", "-mtune=native"

CFFI Empty:
Timings for tabulate calls (n=20) avg: 33.93ms, min: 33.86ms, max: 34.11ms
Timings for element matrix assembly (n=20) avg: 1113.4ms, min: 1102.24ms, max: 1127.39ms

CFFI FFC Code:
Timings for tabulate calls (n=20) avg: 222.15ms, min: 220.06ms, max: 228.79ms
Timings for element matrix assembly (n=20) avg: 1315.5ms, min: 1282.78ms, max: 1340.88ms

CFFI Sparse:
Timings for tabulate calls (n=20) avg: 1263.05ms, min: 896.26ms, max: 1340.22ms
Timings for element matrix assembly (n=20) avg: 2337.94ms, min: 2041.7ms, max: 2405.39ms

CFFI Sparse AVX:
Timings for tabulate calls (n=20) avg: 1418.28ms, min: 1107.83ms, max: 1609.05ms
Timings for element matrix assembly (n=20) avg: 2802.01ms, min: 2666.01ms, max: 3605.4ms

Additionally "-funroll-loops":

CFFI Sparse:
Timings for tabulate calls (n=20) avg: 1081.34ms, min: 945.12ms, max: 1148.13ms
Timings for element matrix assembly (n=20) avg: 2134.41ms, min: 2027.45ms, max: 2206.39ms

CFFI Sparse AVX:
Timings for tabulate calls (n=20) avg: 1277.27ms, min: 1152.35ms, max: 1335.99ms
Timings for element matrix assembly (n=20) avg: 2340.53ms, min: 2122.68ms, max: 2410.81ms