# Input contracts

Liujia CSV is UTF-8 with well_name,depth,RT,GR,DEN,Subfacies. Depth is metres,
RT uses the original ohm-metre convention, GR is API, DEN is g/cm3. Preserve
original row order for source_row identity. Required historical class literals:

| ID | Literal | English meaning |
| --- | --- | --- |
| 0 | 沼泽 | Swamp |
| 1 | 分流河道 | Distributary channel |
| 2 | 河口坝 | Mouth bar |
| 3 | 席状砂 | Sheet sand |
| 4 | 浅湖泥 | Shallow lacustrine mud |
| 5 | 深湖泥 | Deep lacustrine mud |

These operational labels are not independent geological truth. Historical
labels and rule recomputation have different reference roles. No unit inference
or arbitrary category aliases are supported. Original QC rejects RT<=0, GR<=0,
DEN<=0 and DEN>5. Segments break by well, nonpositive depth step, or step >0.075 m.
Context uses the 0.05 m convention. The accepted input has 101,639 rows,
101,615 retained rows, 10 wells, 3 blocks, 14 nonempty segments. Nonfinite
features fail. This is the paper input contract, not a new generic QC protocol.

Group CSV has well_id,spatial_block with one nonmissing assignment per well,
no duplicate wells. Approved groups are inputs; no reclustering or coordinates
are needed. Original functions produce folds and choose calibration wells.
Controlled measurement/group files are excluded from this candidate.

FORCE/KGS units, columns, archive hashes and screening are described in
PUBLIC_WORKFLOWS.md. KGS coarse factors are not Liujia's fixed metre spacings.
Acquire external data directly with applicable permissions.

examples/liuji_input_check.py generates three synthetic wells and groups,
exercises original QC/features with no training, and intentionally fails the
paper-population gate. examples/run_synthetic.py demonstrates pairing and
budgets using a constructed comparator. Neither is field performance evidence.
