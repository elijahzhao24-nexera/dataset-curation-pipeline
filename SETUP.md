# Repo Setup

## 1) Prerequisites

- Python `3.10+`
- `pip` (latest recommended)
- For GPU mode: NVIDIA GPU + compatible NVIDIA driver

## 2) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install dependencies

Choose one path:

```powershell
# CPU path
pip install -r requirements-roboflow-cpu.txt
```

```powershell
# GPU path (current default: CUDA 13.0 wheels)
pip install -r requirements-roboflow-gpu.txt
```

```powershell
# GPU path + optional SAM/SAM3/Gaze/YoloWorld dependencies
pip install -r requirements-roboflow-gpu-extras.txt
```

## 4) What this repo is currently using

Current GPU requirements are pinned to:

- `torch==2.9.1`
- `torchvision==0.24.1`
- `torchaudio==2.9.1`
- PyTorch index: `https://download.pytorch.org/whl/cu130`

Those pins are in:

- `requirements-roboflow-gpu.txt`
- `requirements-roboflow-gpu-extras.txt`

## 5) How to switch CUDA version

To target a different CUDA build, update the first line in both GPU requirements files:

```text
--extra-index-url https://download.pytorch.org/whl/cu130
```

Replace `cu130` with the CUDA build you want (for example `cu128`, `cu126`, `cu124`), then set matching `torch/torchvision/torchaudio` versions for that CUDA build.

## 6) Runtime checks

Quick CUDA + ONNX Runtime provider check:

```powershell
python -c "import torch, onnxruntime as ort; print('torch cuda:', torch.cuda.is_available()); print('ort providers:', ort.get_available_providers())"
```

Expected provider list for GPU includes `CUDAExecutionProvider`.

Check DINOv2 runtime device:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda runtime:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 7) Roboflow preprocessing notes

If you enable Roboflow preprocessing (`USE_BIN_MASK_FOR_EMBEDDING=true`) and see missing SAM/SAM3/Gaze/YoloWorld messages, those are optional model families from `inference`.

If CUDA is not detected and ONNX Runtime only reports CPU providers:

```powershell
pip uninstall -y inference inference-gpu onnxruntime onnxruntime-gpu onnxruntime-directml
pip install -e ".[roboflow-gpu]"
```
