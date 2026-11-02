KGS input preparation

See ../../../docs/PUBLIC_WORKFLOWS.md for current commands, exact archive
hash, source attribution, input units, screening rules and output contracts.

scan_kgs_archive.py --archive INPUT.zip --output-dir OUTPUT
prepare_kgs.py --archive INPUT.zip --inventory OUTPUT/gr_density_inventory.json --output-dir OUTPUT

Both commands support --help without reading input files. Existing generated
outputs are refused. No download or model training is performed by these two
scripts. Original full replay records are retained in the separate author archive.
