# Issue 173
https://bitbucket.org/fenics-project/ffc/issues/173/uflacs-preintegration-emits-inefficient

## Issue

"C++ compiler eats over 10 GB of memory when trying to compile"

## Tracing the issue

Code generation is started by dijitso, in `jit.py`, line 165:
```
header, source, dependencies = generate(jitable, name, signature, params["generator"])
```
Compilation is started in line 176:
```
status, output, lib_filename, err_info = \
                build_shared_library(signature, header, source, dependencies, params)
```
Issue is probably [`generate_preintegrated_dofblock_partition`](https://bitbucket.org/fenics-project/ffc/src/479fae88be777da83742306eefe271cb168fcaa9/ffc/uflacs/integralgenerator.py?at=master&fileviewer=file-view-default#integralgenerator.py-989) method which generates the C++ code for the `tabulate_tensor` function and performs unrolling of 49^2 assignments with many inlined double values.
### Calls stack:
 - Stage 4: code generation in `ffc/compiler.py`, line 200
 - ...
 - `generate_integral_code` called in `ffc/codegeneration.py`, line 249
 -  "Generate code ast for the tabulate_tensor body", calling `IntegralGenerator.generate()` in `ffc/uflacs/uflacsgenerator.py`, line 45
 - "Generate code to compute piecewise constant scalar factors and set A at corresponding nonzero components", calling `generate_preintegrated_dofblock_partition` in `ffc/uflacs/integralgenerator.py`, line 237