# Environment selection

| Chain | Python | NumPy | pandas | scikit-learn | SciPy | joblib | Recorded OS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Liujia main | 3.12.10 | 2.4.3 | 2.3.3 | 1.8.0 | 1.17.1 | 1.5.3 | Windows 11 x64 |
| Liujia matched | 3.13.9 | 2.3.5 | 2.3.3 | 1.7.2 | 1.16.3 | 1.5.2 | Windows 11 x64 |
| FORCE | 3.12.10 (retained author runtime) | 2.4.3 | 2.3.3 | 1.8.0 | 1.17.1 | 1.5.3 | Windows 11 x64 |
| KGS | 3.12.14 | 2.3.5 | 2.2.3 | 1.8.0 | 1.17.0 | 1.5.3 | Linux 6.18.57 x86_64, glibc 2.39 |

Main means class, calibration, phases, triage, and paired. Matched means only the
same-row class/change comparison. Liujia imports matplotlib 3.10.8 (main) or
3.10.6 (matched). FORCE/KGS versions come from their own records, not Liujia.
Original external requirements remain beside those workflows.

The txt files pin observed top-level dependencies, not complete binary builds
or wheel-hash locks. New pip/Conda installation is not numerical acceptance.

Start with the specified Python, then install into separate environments:

```text
python -m venv ../main-env
../main-env/Scripts/python -m pip install -r environments/liuji-main.txt
../main-env/Scripts/python -m pip check
../main-env/Scripts/python workflows/liuji/run.py check-env --environment main
```

Repeat using Python 3.13.9 and liuji-matched.txt. Run check-env with
--environment matched under that interpreter. Put both executable paths in
your local JSON config. Anaconda was the historical matched interpreter
distributor; the dependency versions, not distributor name, define the record.

Alternatively, run from INSIDE the environments directory:

```text
conda env create -f liuji-main.yml
conda env create -f liuji-matched.yml
```

These YAMLs are installation specifications, not historical Conda build exports.
The relative pip requirements resolve from that directory. Do not merge the
two scientific environments. For FORCE use force.txt; for KGS use kgs.txt with
its recorded Linux/Python runtime. POSIX paths use bin/python, not Scripts/python.
Windows interface tests do not certify KGS Linux or cross-platform exact replay.
