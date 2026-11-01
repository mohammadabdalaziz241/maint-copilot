# Running the open-weight comparison on the Surrey GPU machine

The cluster only generates answers. Judging and scoring happen elsewhere with the frozen judge,
so no API key is needed here. Everything is written under `/scratch/ma06314/maint-copilot-cluster`.

1. Copy this folder to the machine (from the Mac, in the project folder):
   `scp -r cluster otter135:/scratch/ma06314/maint-copilot-cluster/`
2. On the machine: `bash /scratch/ma06314/maint-copilot-cluster/cluster/setup.sh`
   (creates the venv on scratch and installs vLLM; about 5-10 minutes, ~10 GB).
3. `bash /scratch/ma06314/maint-copilot-cluster/cluster/probe_models.sh` lists candidate models.
4. Serving, smoke test and the full run are added once a model is chosen
   (see `docs/open_weight_design.md` for the selection rule).
