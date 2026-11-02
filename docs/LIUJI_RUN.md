# Liujia main-result replay

Obtain authorized compiled measurements and fixed spatial groups (INPUT_SCHEMA.md).
No instrument-native file, coordinates, old predictions, temperatures, or author
version directories are needed. Copy workflows/liuji/config/example.json into
a local config. Set input paths, fresh output, and two interpreter paths. JSON
uses forward slashes or escaped backslashes. Relative paths resolve against the
config directory, not the shell directory. Executable names resolve on PATH.
Set protocol to the bundled paper_protocol.json. Altered scientific parameters
are refused: data paths are configurable, the accepted design remains fixed.

```text
python workflows/liuji/run.py plan --config "../my config.json"
python workflows/liuji/run.py inspect --config "../my config.json"
python workflows/liuji/run.py run --config "../my config.json" --allow-training
```

plan validates paths/protocol and prints commands without writing or fitting.
inspect reads input, runs original QC/features/folds and checks the paper
population without fitting. run first verifies both environments, then executes
class, calibration, phases, triage, paired and matched. Configurable per-stage
timeout defaults to one day. After failure preserve logs and use a fresh output.
The internal _stage command is not a resume or tuning interface.

The wrapper stages only measurement/group inputs. Before each stage, source,
config and runtime hashes must match the previous completed checkpoint. Existing
outputs, stage-order errors or foreign caches fail. Old prediction/model paths
are not config options. Logs include input identities and runtime reads and may
contain controlled well identifiers: do not publish the runtime directory.

Internal runtime/ paths retain legacy relative Unicode names to preserve source
I/O contracts, not to require former author folders. Newly generated intermediates
can be reused by downstream stages. Standalone legacy main() functions are not
supported public entries. Use the wrapper and RESULT_ENTRY_INPUT_MAP.csv.

Optional historical comparison is separate and read-only:

```text
python tools/compare_arrays.py "../baseline/result.npz" "../new/result.npz"
```

Comparison checks values/shapes/dtypes, not compressed-container bytes. It does
not automatically certify new results. S26 acceptance scripts and logs remain
an author-held historical record, not normal computation dependencies.
