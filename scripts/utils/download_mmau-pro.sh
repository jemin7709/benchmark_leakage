uv run hf download gamma-lab-umd/MMAU-Pro --repo-type dataset
cd ${HF_HOME:-${HOME}/.cache/huggingface}/hub/datasets--gamma-lab-umd--MMAU-Pro/snapshots/*/
unzip -n ./data.zip || true
rm -f ./data.zip || true