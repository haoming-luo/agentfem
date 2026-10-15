# Heat campaign

Create and run with the installed software:

```sh
agentfem check --runtime
agentfem run --name baseline -- --samples 6
agentfem show latest --files
```

This runs six actual steady heat finite-element models and writes a dataset
manifest plus numerical arrays. No AI or training framework is needed.
The hot-wall temperature varies, the cold wall stays at 300 K, and other
boundaries are insulated. The output is a scalar dataset, not field training
data or a trained surrogate. Its simple maximum-temperature check is not
general physical validation.

Repeat the same command to reuse completed cases from `campaign-data/`.
Each CLI run publishes a separate dataset snapshot under `outputs/`.
Keep the ledger for resume; do not edit it to force incompatible reuse.
For a different scientific experiment use a new project or deliberately
choose a different ledger directory. Preserve source/configuration with data.

First use one process. This template is not an example of across-case process
parallelism. Use the existing Campaign execution policies for larger studies;
do not nest process workers inside a multi-rank solve.
