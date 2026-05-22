# Data

This directory contains the dataset loader used by `train_latentrxnflow.py`
and `eval_multigpu.py`.

Large preprocessed datasets are intentionally not tracked. Place local pickle
files under this directory or pass absolute paths in the config. Each pickle
should contain a list of reaction dictionaries compatible with
`USPTOReact2MainProduct` in `uspto_main_product.py`.

The loader expects the usual NERF reaction fields such as `element`,
`src_bond`, `tgt_bond`, `src_aroma`, `tgt_aroma`, `src_charge`, `tgt_charge`,
`src_mask`, `tgt_mask`, `src`, `tgt`, and token fields. Conditional models also
expect condition features such as `condition_fp` and `condition_num`.
