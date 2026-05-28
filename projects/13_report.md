---
layout: page
title: CP 330 - Edge AI
subtitle: Indian Institute of Science | January 2025
---
# Nighttime Street Light Functional Audit Using Computer Vision on Edge Device

## 1. Problem Statement, Motivation & Objectives (1–2 paragraphs + 3–5 bullets)
Urban street lighting is a critical component of public infrastructure it improves visibility, reduces crime, and supports safe vehicular and pedestrian movement after dark. Yet maintaining expansive street lighting networks at city scale remains a formidable challenge. Cities worldwide operate hundreds of thousands of streetlights (Los Angeles alone maintains over 220,000 units), and globally the number is projected to exceed 350 million. Despite their importance, most cities still rely on outdated maintenance practices: periodic manual inspections and citizen complaint-driven reporting. These approaches are labor-intensive, error-prone, and slow to detect outages, leaving malfunctioning lamps unnoticed for extended periods and compromising both public safety and energy efficiency.

The motivation for an automated, vision-based approach is clear: IoT sensor networks that instrument every lamp post carry prohibitive deployment and maintenance costs, while manual drive-by surveys lack scientific reliability and scalability. Computer vision offers a compelling middle path using cameras mounted on ordinary vehicles to perform continuous nighttime drive-by surveys, turning everyday city fleets into infrastructure monitoring platforms. 

Kumar et al. (2016) demonstrated the feasibility of this paradigm using a car-top sensor platform with cameras, lux sensors, GPS, and IMU, achieving lamp identification and height estimation across multiple cities. Building on this foundation, recent deep learning approaches have substantially improved detection accuracy — particularly for small, distant, or partially occluded lamps under challenging nighttime conditions. Edge AI deployment of such models is well-motivated: processing video locally on the vehicle eliminates the latency and bandwidth costs of cloud transmission, enables real-time fault flagging, and avoids transmitting sensitive location-linked footage off-device.

### Objectives
1. Develop a robust detection model capable of handling nighttime, low-light, and motion-blurred imagery.  

2. Implement an image preprocessing pipeline (CLAHE, Gamma correction) to enhance nighttime visibility.  

3. Optimize models via pruning and quantization for deployment on resource-constrained hardware.  

4. Deploy a real-time web-based interface for live monitoring.

---

## 2. Proposed Solution (Overview)

The system is a vehicle-mounted, edge-deployed streetlight fault detector. A camera captures nighttime video during drive-by patrols; the webapp runs a compressed YOLO model to classify each detected lamp as ON or OFF in real time, logging faults with timestamps for subsequent municipal review — with no cloud connectivity required.

**End-to-end pipeline:**

**Data** → Nighttime video is captured along pre-planned routes and extracted into frames. Each frame is labelled with bounding boxes for `ON_Streetlight` and `OFF_Streetlight` classes. A preprocessing pipeline (CLAHE, gamma correction, unsharp masking) is applied to compensate for low ambient light and motion blur before training.

https://drive.google.com/drive/folders/1SJH9nexL3rW9DqcZN_Q-33obRFqGnCKH?usp=sharing


**Model** → Multiple full-precision (FP32) baseline models are trained on a GPU workstation (Google Colab, Tesla T4): YOLOv5nu, YOLOv5su, YOLOv8n, and YOLOv8s. The best-performing lightweight model (YOLOv5nu) is selected for compression.

**Compression** → Post-training quantization (ONNX Runtime dynamic INT8 and FP16), one-shot and iterative magnitude pruning (detection head excluded to preserve accuracy), and knowledge distillation (YOLOv5su → YOLOv5nu and YOLOv8s → YOLOv8n pairs) are applied to reduce model size and inference latency to levels compatible with mobile-based webapp hardware.

**Deployment** → The compressed model is exported to ONNX (opset 17) and TFLite INT8 formats (input resolution 320×320) and deployed on a mobile-based webapp. On-device inference time is benchmarked using ONNX Runtime on CPU with threads pinned to mimic Raspberry Pi 4, resource utilisation, and detection quality are measured and reported.

**Output** → Per-frame bounding box predictions with class labels (ON / OFF) and confidence scores, rendered in real time and logged for fault mapping.

---

## 3. Hardware & Software Setup

**Hardware:**

- **Data collection:** Smartphone camera mounted on vehicle dashboard, used for nighttime drive-by footage capture across three Bengaluru localities
- **Training platform:** Google Colab with NVIDIA Tesla T4 GPU (14,913 MiB VRAM)
- **Edge deployment platform:** Smartphone based Webapp (target device for TFLite INT8 / ONNX inference)

**Software:**

| Component | Tool / Version |
|---|---|
| Manual Annotation | LabelImg |
| Training framework | Ultralytics 8.4.45 (YOLOv5/v8), PyTorch 2.10.0+cu128 |
| Baseline object detection | YOLOv5nu, YOLOv5su, YOLOv8n, YOLOv8s |
| Preprocessing | OpenCV (CLAHE, gamma, unsharp), Albumentations |
| Compression | PyTorch pruning (`torch.nn.utils.prune`, head-excluded); ONNX Runtime dynamic INT8; FP16 export |
| Knowledge distillation | Ultralytics KD trainer with `model.loss` wrap (α=0.5, T=4, AdamW, cosine LR); pairs: YOLOv5su→YOLOv5nu, YOLOv8s→YOLOv8n |
| Export | Ultralytics ONNX export (opset 17, `simplify=True`), TFLite INT8 (imgsz 320) |
| Edge runtime | TFLite runtime / ONNX Runtime on Mobile Phone; ORT CPU benchmark (4 threads) as RPi 4 proxy |
| Development | Python 3.12.13, Google Colab, Jupyter Notebook |

---

## 4. Data Collection & Dataset Preparation

**Data source:** Custom dataset — nighttime video collected by the project team during vehicle drive-bys across three localities in Bengaluru, Karnataka, India.

**Collection routes and distances:**

| Area | Route Distance |
|---|---|
| IISc Campus | 4,564.4 m |
| Mathikere | 3,679.3 m |
| Yesvantpur | 8,313.9 m |
| **Total** | **16,557.7 m (~16.6 km)** |

The dataset preparation followed a specific downsampling and annotation pipeline:

Frame Rate: Video data was originally captured at 30fps.  

Annotation Sampling: To create the training set, frames were annotated at a reduced rate of 2fps.  

Tools: Manual labeling and bounding box creation were performed using LabelImg.  

Final Dataset: The resulting processed dataset consists of ~ 2.1k frames ready for model training.

**Dataset characteristics:**
- **Classes:** 2 — `ON_Streetlight` (functioning lamp), `OFF_Streetlight` (faulty / unlit lamp)
- **Class imbalance:** The OFF class is significantly underrepresented (~11% of instances), reflecting real-world fault rates
- **Training/validation/test split:** ~70% / 20% / 10%
- **Image resolution:** 640×640 (resized during training); exported at 320×320 for edge deployment

**Imaging conditions:** All footage is nighttime, low ambient light, with motion blur from vehicle movement and handheld capture. These conditions make detection significantly harder than standard benchmark datasets.

**Preprocessing pipeline (applied before training):**

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation, clip=2.0, tile=8×8) restores local contrast suppressed by low ambient light
2. **Gamma correction** (γ=1.4) brightens underexposed regions without saturating lamp blobs
3. **Unsharp mask** (strength=1.3, σ=3) recovers edges softened by motion blur

**Training augmentations (Albumentations):**
- Random rotation ±20°, horizontal flip (p=0.5)
- Random brightness/contrast ±30% (p=0.6)
- Gaussian blur (kernel 3–5, p=0.3)
- CLAHE (p=0.4), random gamma 80–120 (p=0.4)
- Mosaic (1.0), random erasing (0.3), HSV value shift (0.4)

**Labelling:** Bounding box annotations created manually; each frame is tagged with class ID, centre coordinates, and box dimensions in YOLO format.

---

## 5. Model Design, Training & Evaluation

**Architecture overview:**

All YOLO models follow the standard one-stage detection paradigm: a CNN backbone extracts multi-scale feature maps, a neck (Feature Pyramid Network) fuses them, and detection heads predict bounding boxes and class probabilities at three scales. Models differ in depth and width multipliers, giving a range of parameter counts and computational costs.

**Models trained and their specifications:**

| Model | Params | GFLOPs | Layers | Weight Size |
|---|---|---|---|---|
| YOLOv5nu | 2,503,334 | 7.1 | 85 | 5.3 MB |
| YOLOv5su | 9,112,310 | 23.8 | 85 | 18.5 MB |
| YOLOv8n | 3,006,038 | 8.1 | 73 | 6.2 MB |
| YOLOv8s | 11,125,971 | 28.4 | 225 | 22.5 MB |


**Training configuration (YOLO models):**

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Image size | 640×640 |
| Batch size | 16 |
| Learning rate (lr0) | 0.01 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| LR schedule | Cosine annealing |
| Bounding box loss | CIoU |
| Objectness/class loss | Binary cross-entropy |
| Hardware | Tesla T4 GPU (Google Colab) |

**Evaluation results (validation set):**

| Model | Precision | Recall | mAP50 | F1 | Inference (GPU) |
|---|---|---|---|---|---|---|
| YOLOv5nu | 0.833 | 0.826 | 0.886 | 0.829 |
| YOLOv5su | 0.856 | 0.809 | 0.862 | 0.832 |
| YOLOv8n | 0.833 | 0.803 | 0.877 | 0.817 |
| YOLOv8s | 0.858 | 0.808 | 0.878 | 0.832 |


**Key finding:** YOLOv5nu achieves the best balance of accuracy and speed, highest mAP50 (0.886), smallest parameter count (2.5M), and fastest GPU inference (3.2ms). 
---

## 6. Model Compression & Efficiency Metrics

Four compression strategies were applied to produce edge-ready models, targeting constrained compute and memory budgets on mobile and Raspberry Pi 4 class hardware. The pipeline corrects several methodological issues present in earlier versions (dynamic PyTorch INT8 on Conv2d is a no-op; RPi FPS now measured via ORT CPU benchmarking, not a formula estimate; KD loss is injected by wrapping `model.loss` rather than dead-code trainer overrides; the detection head is excluded from pruning to prevent mAP collapse).

**Strategy 1 — Post-Training Quantization: FP16 and ONNX Runtime Dynamic INT8**

Two honest, deployable quantization paths are used:

- **FP16 (half-precision):** The FP32 PyTorch model is exported to ONNX with half-precision weights. Provides ~2× size reduction with negligible accuracy loss on GPU/ARM hardware with native FP16 support.
- **ONNX Runtime Dynamic INT8:** The ONNX model is quantized to INT8 using `onnxruntime.quantization.quantize_dynamic` with `QuantType.QInt8`. Unlike PyTorch's `torch.quantization.quantize_dynamic`, ORT dynamic INT8 operates correctly on Conv2d layers, making it the valid post-training INT8 path for YOLO backbones. Provides ~4× size reduction.
- Applied to: all four YOLO variants (YOLOv5nu, YOLOv5su, YOLOv8n, YOLOv8s)

**Strategy 2 — One-Shot Magnitude Pruning**

L1 unstructured pruning removes the lowest-magnitude weights from Conv2d layers in a single pass. Applied to all four YOLO variants at sparsity levels of 20%, 30%, and 50%. The final `prune_skip_last_n_convs=6` Conv2d layers (the Detect head's 1×1 convs) are excluded — pruning these produces large mAP drops for negligible size gain. Pruned models report both raw file size and gzipped file size; the gzip size is the realistic deployable size, as zero runs compress effectively.

- Applied to: YOLOv5nu, YOLOv5su, YOLOv8n, YOLOv8s
- Sparsity levels: 20%, 30%, 50% of backbone Conv2d weights zeroed
- Reported size: raw `.pt` file size and gzip-compressed size (`FileSizeMB_gz`)

**Strategy 3 — Iterative Magnitude Pruning**

Progressive pruning in five steps (10% → 20% → 30% → 40% → 50%) with 5-epoch fine-tuning between each step. Crucially, prune masks are kept attached during fine-tuning — an earlier version removed masks between steps, allowing zeroed weights to regrow and invalidating the sparsity budget. The detection head is excluded here as well.

- Applied to: all four YOLO variants
- Steps: 5 rounds × 5 fine-tuning epochs
- Final sparsity: ~50% backbone Conv2d weights removed; masks held in place throughout fine-tuning

**Strategy 4 — Knowledge Distillation**

A larger teacher model supervises a smaller student model via a weighted loss combination. The KD loss is injected by wrapping `model.loss` on the student — an earlier approach that overrode `train_step` was dead code never called by the Ultralytics `BaseTrainer`. Two teacher→student pairs are trained:

| Pair | Teacher | Student |
|---|---|---|
| v5s→v5n | YOLOv5su | YOLOv5nu |
| v8s→v8n | YOLOv8s | YOLOv8n |

> Loss = α · L_KD + (1−α) · L_task, α=0.5, Temperature T=4

- Optimizer: AdamW (lr=1e-3, weight decay=1e-4)
- Schedule: Cosine annealing, patience=5 early stopping
- Goal: Transfer knowledge from the larger model to a model with fewer parameters at no extra data cost

**Summary of compressions:**

| Tag | Category | mAP50 | F1 | FileSizeMB | FileSizeMB_gz | RPi_FPS |
|---|---|---|---|---|---|---|
| yolov5nu | Baseline | 0.8859 | 0.8294 | 5.25 | 4.74 | 5.7 |
| yolov5nu_fp16 | Quantized-FP16 | 0.8859 | 0.8290 | 5.14 | 4.69 | 4.1 |
| yolov8s | Baseline | 0.8782 | 0.8322 | 22.50 | 20.74 | 1.9 |
| yolov8s_fp16 | Quantized-FP16 | 0.8781 | 0.8328 | 22.37 | 20.67 | 1.9 |
| yolov8n_fp16 | Quantized-FP16 | 0.8775 | 0.8175 | 6.13 | 5.62 | 3.5 |
| yolov8n | Baseline | 0.8774 | 0.8174 | 6.23 | 5.66 | 4.6 |
| yolov5su_pruned_20pct | Pruned-OneShot | 0.8636 | 0.8266 | 36.62 | 27.59 | 2.2 |
| yolov5su | Baseline | 0.8625 | 0.8316 | 18.50 | 17.02 | 2.3 |
| yolov5su_fp16 | Quantized-FP16 | 0.8623 | 0.8317 | 18.36 | 16.96 | 2.1 |
| yolov5nu_pruned_20pct | Pruned-OneShot | 0.8284 | 0.7725 | 10.18 | 7.80 | 4.7 |
| yolov5su_pruned_30pct | Pruned-OneShot | 0.7788 | 0.7082 | 36.62 | 25.37 | 2.2 |
| yolov8n_pruned_20pct | Pruned-OneShot | 0.7435 | 0.7209 | 12.18 | 9.33 | 3.5 |
| yolov8s_pruned_20pct | Pruned-OneShot | 0.5152 | 0.5531 | 44.66 | 33.50 | 1.9 |
| yolov8n_pruned_30pct | Pruned-OneShot | 0.4385 | 0.4676 | 12.18 | 8.53 | 4.6 |
| yolov5nu_pruned_30pct | Pruned-OneShot | 0.3951 | 0.4473 | 10.18 | 7.15 | 4.1 |
| yolov8s_pruned_30pct | Pruned-OneShot | 0.2206 | 0.2490 | 44.66 | 30.72 | 1.9 |
| yolov5su_pruned_50pct | Pruned-OneShot | 0.0045 | 0.0152 | 36.62 | 20.32 | 2.2 |
| yolov5nu_pruned_50pct | Pruned-OneShot | 0.0013 | 0.0050 | 10.18 | 5.71 | 6.0 |
| yolov8s_pruned_50pct | Pruned-OneShot | 0.0000 | 0.0000 | 44.66 | 24.50 | 1.9 |
| yolov8n_pruned_50pct | Pruned-OneShot | 0.0000 | 0.0000 | 12.18 | 6.77 | 3.5 |
| yolov5nu_ort_int8 | Quantized-ORT-INT8 | NaN | NaN | 2.75 | 1.94 | 10.8 |
| yolov5nu_tflite_int8 | Quantized-TFLite | NaN | NaN | 2.71 | 2.26 | NaN |
| yolov5su_ort_int8 | Quantized-ORT-INT8 | NaN | NaN | 9.38 | 6.45 | 5.1 |
| yolov5su_tflite_int8 | Quantized-TFLite | NaN | NaN | 9.39 | 7.99 | NaN |
| yolov8n_ort_int8 | Quantized-ORT-INT8 | NaN | NaN | 3.23 | 2.28 | 10.8 |
| yolov8n_tflite_int8 | Quantized-TFLite | NaN | NaN | 3.20 | 2.69 | NaN |
| yolov8s_ort_int8 | Quantized-ORT-INT8 | NaN | NaN | 11.36 | 7.83 | 4.4 |
| yolov8s_tflite_int8 | Quantized-TFLite | NaN | NaN | 11.39 | 9.68 | NaN |
---

## 7. Model Deployment & On-Device Performance

**Deployment steps:**

1. **Select models for deployment:** YOLOv8n (best mAP50-95) was flagged for Mobile Phone export
2. **Export YOLO models:**
   ```
   model.export(format="onnx", imgsz=320, simplify=True, opset=17)
   model.export(format="tflite", imgsz=320, int8=True)
   ```
3. **Quantize ONNX for INT8:** Apply ORT dynamic INT8 quantization (`quantize_dynamic`, `QuantType.QInt8`) to the exported ONNX file
4. **Transfer to Mobile phone and use web interface:** Copy .onnx files; run ONNX file in web insterface
5. **Run inference:** Feed preprocessed (CLAHE → gamma → unsharp) frames through the runtime at 320×320 resolution; parse bounding box outputs and render detections

**GPU performance (GPU reference — Tesla T4):**

| Model | Preprocess | Inference | Postprocess | Total per frame |
|---|---|---|---|---|
| YOLOv5nu | 1.8 ms | 3.2 ms | 2.2 ms | ~7.2 ms |
| YOLOv5su | 1.8 ms | 4.9 ms | 2.0 ms | ~8.7 ms |
| YOLOv5mu | 1.8 ms | 11.0 ms | 2.4 ms | ~15.2 ms |
| YOLOv8n | 1.8 ms | 3.4 ms | 2.0 ms | ~7.2 ms |

**Real-time behaviour:** The deployed system processes each incoming frame, draws bounding boxes labelled ON_Streetlight or OFF_Streetlight with confidence scores, and logs detections to a timestamped GPS CSV for post-route fault mapping.

---

## 8. Conclusions & Limitations

**Key outcomes:**

- A custom nighttime streetlight detection dataset was successfully collected across ~16.6 km of Bengaluru roads, covering IISc Campus, Mathikere, and Yeshwantpur
- YOLOv5nu achieved the best overall efficiency profile: highest mAP50 (0.886) at the smallest parameter count (2.5M, 5.3 MB), with GPU inference of 3.2 ms/frame.
- YOLOv8n was competitive (mAP50=0.877) with similar speed and slightly better mAP50-95 (0.458 vs 0.438)

- Quantization (ORT dynamic INT8 and FP16), pruning (one-shot Magnitude)

- RPi 4 inference speed is measured via real ORT CPU benchmarks (4 threads pinned) rather than a formula estimate, giving realistic deployable performance figures

**Limitations:**

- **Class imbalance:** Very less OFF_Streetlight instances in the validation set; model robustness on diverse fault types (partial occlusion, dimming, intermittent faults) is not fully validated
- **Single city, single season:** Data was collected in Bengaluru under dry-season nighttime conditions; performance under rain, fog, or different lamp architectures is unknown

---

## 9. Future Work

- **Dataset expansion:** Collect additional OFF_Streetlight examples across rain, fog, and different lamp types to improve class balance and generalisation

- **Fault severity grading:** Extend the label set beyond binary ON/OFF to include partial fault, flickering, and dim lamp classes for more actionable maintenance prioritisation

- **Fleet-scale deployment:** Integrate the system into existing municipal maintenance vehicles (BBMP, BESCOM fleet) to survey the full city lighting network on routine patrol routes

---

## 10. Challenges & Mitigation

| Challenge | Description | Mitigation |
|---|---|---|
| **Low-light image quality** | Nighttime frames are noisy, motion-blurred, and underexposed, making features difficult to learn | Applied CLAHE + gamma correction + unsharp masking as a fixed preprocessing pipeline before both training and inference |
| **Severe class imbalance** | OFF_Streetlight instances are ~8× less frequent than ON, risking the model ignoring faulty lamps | Mosaic and random erasing augmentations increase exposure to rare class instances; model still learned high recall for OFF class |
| **Edge export resolution trade-off** | Full 640×640 inference is too slow for Mobile Phones; reducing to 320×320 risks missing small distant lamps | Exported all edge models at 320×320; validated that mAP remains acceptable at reduced resolution before committing |
| **GPU-to-edge latency gap** | Tesla T4 inference times (3–11 ms) are not representative of Raspberry Pi performance | RPi 4 speed is now proxied by real ORT CPU benchmarking with 4 threads pinned (mimicking RPi 4 cores)|
| **Data collection logistics** | Nighttime driving routes required planned sorties across three localities with consistent camera placement | Used a fixed dashboard mount and pre-planned OSM-exported routes to ensure route reproducibility and coverage |

---

## 11. References

1. Kumar, S., Deshpande, A., Ho, S.S., Ku, J.S., & Sarma, S.E. (2016). *Urban Street Lighting Infrastructure Monitoring Using a Mobile Sensor Platform.* IEEE Sensors Journal, 16(12), 4981–4994. https://doi.org/10.1109/JSEN.2016.2552249

2. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* arXiv:1506.02640.

3. Ultralytics. (2024). *YOLOv5 and YOLOv8 Documentation.* https://docs.ultralytics.com

4. Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks.* Proceedings of CVPR, 7132–7141.

5. Howard, A. et al. (2019). *Searching for MobileNetV3.* arXiv:1905.02244.

6. Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* Proceedings of CVPR.

7. Han, S., Pool, J., Tran, J., & Dally, W. (2015). *Learning both Weights and Connections for Efficient Neural Networks.* NeurIPS.

8. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.

9. TensorFlow Lite. (2024). *Post-Training Quantization.* https://www.tensorflow.org/lite/performance/post_training_quantization

10. Buslaev, A. et al. (2020). *Albumentations: Fast and Flexible Image Augmentations.* Information, 11(2), 125.

11. OpenStreetMap contributors. (2024). *Bengaluru road network data.* https://www.openstreetmap.org
