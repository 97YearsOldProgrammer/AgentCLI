#!/bin/bash

#################### Local GPU Inference Setup ####################

echo "=== Local GPU Inference Setup ==="
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. GPU drivers may not be installed."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

#################### Option 1: vLLM (Recommended) ####################

setup_vllm() {
    echo "Installing vLLM..."
    pip install vllm --break-system-packages
    
    echo ""
    echo "To start vLLM server, run:"
    echo "  vllm serve nvidia/Llama-3.1-Nemotron-Nano-8B-v1 --port 8000"
    echo ""
    echo "Or with specific GPU memory:"
    echo "  vllm serve nvidia/Llama-3.1-Nemotron-Nano-8B-v1 --port 8000 --gpu-memory-utilization 0.8"
}

#################### Option 2: Ollama (Easiest) ####################

setup_ollama() {
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    echo "Pulling nemotron-mini model..."
    ollama pull nemotron-mini
    
    echo ""
    echo "To start Ollama server, run:"
    echo "  ollama serve"
}

#################### Menu ####################

echo "Select inference backend:"
echo "  1) vLLM (fastest, needs ~16GB VRAM)"
echo "  2) Ollama (easiest, auto memory management)"
echo ""
read -p "Choice [1/2]: " choice

case $choice in
    1) setup_vllm ;;
    2) setup_ollama ;;
    *) echo "Invalid choice" ;;
esac