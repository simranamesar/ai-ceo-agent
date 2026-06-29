#!/bin/bash
set -e

# always run from the project root (this script's folder)
cd "$(dirname "$0")"

# --- 1. stop anything left running from a previous start (prevents NFS "busy") ---
pkill -f streamlit    || true
pkill -f cloudflared  || true
pkill -f run_pipeline || true
sleep 1

# --- 2. virtualenv: reuse if present, build only when missing ---
if [ ! -d .venv ]; then
  echo "[setup] creating .venv ..."
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  # pin the protobuf runtime so ChromaDB / OpenTelemetry stay on 5.x
  pip install --force-reinstall --no-deps "protobuf==5.29.5"
  touch .deps_ok
else
  source .venv/bin/activate
  # only reinstall if requirements.txt changed since last successful install
  if [ ! -f .deps_ok ] || [ requirements.txt -nt .deps_ok ]; then
    echo "[setup] requirements changed — installing missing packages ..."
    pip install -r requirements.txt --quiet
    pip install --force-reinstall --no-deps "protobuf==5.29.5" --quiet
    touch .deps_ok
  else
    echo "[setup] packages up to date, skipping install"
  fi
fi
python -c "import sys; print('venv:', sys.prefix)"                 # should end in /.venv
python -c "import google.protobuf as p; print('protobuf', p.__version__)"

# --- 3. keep TF installed but never let transformers import it ---
#     (the protobuf gencode/runtime clash only happens via the TF proto path)
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# If the protobuf error STILL appears, remove TF (project is torch-only, safe):
# pip uninstall -y tensorflow tf-keras keras tensorboard

# --- 4. model backend (in-process on the GPU) ---
export LLM_PROVIDER=local
export LLM_MODEL=Qwen/Qwen3-8B

# --- 5. build the knowledge base + analysis ---
python -m pipeline.run_pipeline

# --- 6. launch the dashboard, then expose it via the tunnel ---
streamlit run dashboard/app.py --server.port 8501 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false &
sleep 5
./cloudflared tunnel --url http://localhost:8501

# To force a clean rebuild later: stop this script, then `rm -rf .venv` and re-run.
