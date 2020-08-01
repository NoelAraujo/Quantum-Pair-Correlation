Quantum Pair Correlation

Implementation into Julia of 2nd order terms in 'Cooperative Mollow: equations for baby-Mollow with truncation' manuscript - see src folder.

The goal is to find the scaling of saturation of field with subradiant population and lifetimes.
One may change some simulation's parameters in Input Section in files "first/second/third_script_to_run.jl".

To explore GPU's optimization, the core computation uses ArrayFire.jl package. Therefore, the number of particles should be large enough to compensate for overheads. From my experience, N=50 and above.

For now, no extra documentation is provided, because files are directed for specialists in the area. Hence, codes are available for reproducibility purposes.


Keywords :  Bogoliubov–Born–Green–Kirkwood–Yvon hierarchy, BBGKY hierarchy, Julia language