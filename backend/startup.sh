#!/bin/bash
echo "=== SCADA/ICS Security System Starting ==="

# Train ML models if not exists
if [ ! -f "models/rf_bs_tuned.pkl" ]; then
    echo "Training ML models (first run)..."
    python train.py
    echo "ML training complete!"
else
    echo "ML models found, skipping training."
fi

echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
