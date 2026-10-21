# CLIP INT8/FP16 TensorRT + C++ road-scene classifier

A reproducible **workflow** for selective INT8 post-training quantization of CLIP ViT-B/16, FP16 TensorRT comparison, and standalone C++ road-scene classification (daytime/night/dawn-dusk). The model is pretrained and uses three fixed zero-shot CLIP text prompts; no task-specific training is performed.

## Previously validated results (Tesla T4, original project run)

| Metric | FP32 PyTorch | FP16 TensorRT | Selective INT8/FP16 TensorRT |
| --- | ---: | ---: | ---: |
| Balanced 300-image accuracy | 81.00% | 80.67% | 81.33% |
| Engine size | — | 166.14 MiB | 114.42 MiB |
| Batch-1 mean GPU execution | — | 4.410 ms | 4.229 ms |
| Batch-32 mean GPU execution | — | 91.727 ms | 84.006 ms |
| Batch-32 throughput | — | 348.9 img/s | 380.9 img/s |

Independent **C++** JPEG-to-prediction implementation: 80.67% accuracy, 33.77 ms mean end-to-end latency (24.80 ms CPU preprocessing, 8.71 ms GPU timing), 32.56 ms median, 39.96 ms p95, measured on 300 locally stored images with a persistent engine. Do **not** substitute the 4.23 ms back-to-back CUDA benchmark for C++ application GPU latency. Original validated figures are reference observations, *not guaranteed reproduction targets*.

Dawn/dusk recall was approximately 49–51%; predictions and cosine scores are not calibrated probabilities. The original 300-image development set was balanced and small.

## Quick start: Google Colab with T4 GPU

1. Create a Google Colab notebook, select **Runtime → Change runtime type → T4 GPU**. NVIDIA GPU, Linux, and an accessible dataset repository are required.
2. Clone this repository (replace the URL with your actual repository URL):

   ```bash
   !git clone https://github.com/YOUR_USERNAME/clip-tensorrt-cpp-inference.git /content/clip-tensorrt-cpp-inference
   ```
3. Open `notebooks/run_pipeline.ipynb` in Colab and run all cells. Google Drive consent is required. The notebook installs requirements once and invokes the stage runner.
4. Re-running the workflow skips hash-verified artifacts. An interrupted stage is rebuilt.

Alternatively, after installing `requirements.txt` and mounting Drive:

```bash
cd /content/clip-tensorrt-cpp-inference
python -m src.pipeline --stage all --root /content/drive/MyDrive/CLIP_Quantization_Project
```

To run a single step, pass `--stage prepare|baseline|quantize|convert|build|evaluate`; prerequisites from earlier stages must already exist. The first run downloads 600 labeled JPEGs (300 calibration, 300 validation), the pretrained CLIP checkpoint, and creates the checkpoints/ONNX graphs/engines locally before copying their verified bytes to Drive. Subsequent runs use valid stage outputs.

The project intentionally keeps Hugging Face caches in `/content/hf`, **not Google Drive**, avoiding reuse of corrupt cached PyTorch `.bin` downloads; `use_safetensors=True` is explicitly requested. Data revision is pinned. No images or pretrained weights are redistributed in this repository; follow BDD100K and upstream model terms.

### Artifacts and reproducibility

`src/pipeline.py` is the single source of truth. It uses deterministic sampling (seed 42 for calibration, 43 for evaluation), emits saved manifests, performs 300-image FP32 evaluation, calibrates 123 enabled quantizers with NVIDIA ModelOpt (max calibration), exports 123 ONNX Q/DQ pairs, restores 246 FP32 quantizer scales, confirms **zero changes over 57,364,224 quantized weight integers**, builds FP16 and mixed TensorRT engines (batch min=1, opt=8, max=32), and evaluates both on exactly the same validation manifest. Output checksums live in `<root>/checksums/`. `src/common.py` handles verified Drive publication.

ModelOpt `mto.restore` reloads saved calibration when present; no recalibration is needed for valid checkpoints. Engines are deserialized before reuse. A changed Torch/Transformers/driver/TRT stack may require rebuilding engines from ONNX; remove stale outputs from the persistent folder if changing the configuration or dependencies.

**Software caveat:** Core TensorRT 11.4 / ModelOpt 0.47 / ONNX 1.21 versions are pinned, but the environment includes Colab-provided CUDA/PyTorch and dependency compatibility ranges. For forensic exact-environment replay, capture and freeze a successful `pip freeze` and driver/clock info. Hardware availability, remote dataset availability, CUDA tactics, and GPU timing are not controllable by this repository. We have syntax/unit-checked this packaging, **not rerun an end-to-end fresh Tesla T4 pipeline** as part of repository generation. No promise of failure-free execution or numerically identical latency across arbitrary GPUs.

### Build and run C++ classifier

C++ requires TensorRT **11.4 C++ headers**, CUDA development headers/toolkit, OpenCV 4 development package, and a GPU-compatible serialized engine. Python TensorRT wheels may include runtime libraries but not headers. Set `TRT_INCLUDE_DIR` to a matching headers path and optionally `TRT_LIB_DIR` (else `cpp/build.sh` detects the pip-installed TensorRT runtime library):

```bash
sudo apt-get update && sudo apt-get install -y libopencv-dev pkg-config g++
export TRT_INCLUDE_DIR=/path/to/tensorrt-11.4/include
# optional: export TRT_LIB_DIR=/path/to/installed/tensorrt_libs
bash cpp/build.sh
cpp/clip_classifier \
  /path/to/clip_attention_fc1_int8_fp16_repaired.engine \
  /path/to/CLIP_Quantization_Project/cpp_deployment/text_embeddings_fp32.bin \
  /path/to/road_image.jpg
```

The executable emits CSV columns: `index,prediction,s_daytime,s_night,s_dawn_dusk,preprocess_ms,gpu_ms,end_to_end_ms`. Engine load is excluded from per-image timing. The OpenCV decoder and custom antialiased bicubic resize closely approximate CLIP/Pillow; they are not guaranteed bitwise-identical. Evaluate C++ predictions independently rather than assuming PyTorch parity.

### What is deliberately NOT in Git

Large `.engine`, `.onnx`, `.pth`, `*.bin`, `.jpg` files, extracted NVIDIA vendor headers, and compiled `cpp/clip_classifier`. These are generated, hardware-dependent, licensed, or larger than normal GitHub file limits. `text_embeddings_fp32.bin` is generated by the pipeline from the three fixed prompts (6 KiB) and saved in the persistent `cpp_deployment/` folder. The dataset manifests are generated in the persistent workspace; commit copies after verifying dataset redistribution terms.

### Testing

```bash
python -m compileall -q src
python -m pytest -q tests  # pytest installation required
```

These CPU tests check artifact integrity and syntax; they are **not a substitute for the T4 GPU acceptance test**. See `tests/ACCEPTANCE.md`.

### Limitations

- Sampled 300 validation images: accuracy differences of one or two images are not significant evidence of accuracy improvements.
- Important dawn/dusk recall weakness; do not use as a safety-critical driving system.
- C++ preprocessing may change predictions compared with the Python CLIP processor.
- 31% smaller engine is a storage improvement, **not** a 31% reduction in measured GPU peak memory.
- TensorRT engines are not portable across arbitrary drivers/GPUs, and exact timing cannot be promised.
