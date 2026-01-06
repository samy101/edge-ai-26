---
layout: page
title: CP 330 - Edge AI
subtitle: Indian Institute of Science | January 2025
---

# Portable AI Sign-Language Translator

## Introduction

Millions of deaf and hard-of-hearing individuals face daily communication barriers in environments like hospitals, banks, government offices, and public spaces. These challenges arise primarily because most people do not understand sign language, limiting effective and inclusive communication. 

The goal of this project is to bridge this communication gap by developing a portable, real-time AI-based sign language translator using a camera-equipped microcontroller. The system captures hand gestures, classifies them using a trained machine learning model, and translates them into text (and optionally voice) to facilitate interaction between sign language users and non-signers.

## Dataset

The signs used in this project were sourced from the publicly available Indian Sign Language Detection dataset hosted on Roboflow Universe. The dataset comprises RGB images featuring a single individual performing various everyday Indian Sign Language (ISL) gestures. Each image is annotated with a precise hand-region bounding box and a corresponding single-class label. For this work, we selected the 15 most essential static signs to enable the final model to support basic conversational interactions, including greetings, personal needs, and common requests.

### Dataset Statistics

- Total images: 3,036
- Distinct classes: 15 ISL signs
- Average images per class: ≈ 202
- Bounding boxes: One per image (signing hand)

### Table 1: Per-class image distribution in the cleaned dataset subset

| **Class** | **# Images** | **Class** | **# Images** |
|-----------|--------------|-----------|--------------|
| agree     | 201          | hungry    | 202          |
| angry     | 203          | me        | 201          |
| bad       | 202          | please    | 202          |
| come      | 201          | sorry     | 200          |
| fine      | 216          | thank     | 201          |
| go        | 201          | you       | 201          |
| happy     | 201          | hello     | 201          |
| how       | 203          | --        | --           |
| **Total** |              |           | **3,036**    |


## Model Architecture & Training
We initially developed a custom-trained MobileNetV2-based object detection model. However, due to deployment limitations on the Nicla Vision, we later transitioned to using Edge Impulse Studio, which provided better optimization for embedded deployment. The complete Edge Impulse flow (including model architecture and deployment pipeline) is included in a separate folder within this repository.

### Custom Model Details

Base Model
- **MobileNetV2** (α = 0.35) backbone (ImageNet pretrained, top removed)
- **Lightweight head:** Dense(128, ReLU) → Dropout(0.3) → Softmax(15 classes)

Training Strategy
- **Stage 1:** Freeze backbone, train head (5 epochs)
- **Stage 2:** Unfreeze last 30 layers, fine-tune whole model at lower LR (5 epochs)

## Experiments and Results

### Baseline CNN Prototype

**Pre-processing:**  
All input images were resized to `96×96` Greyscale and normalized to the range `[0,1]`. During training, simple data augmentation was applied—random horizontal flips and small in-plane rotations up to ±15°.

**Network architecture:**  
We implemented a basic 5-layer CNN as the initial prototype. The network consists of three convolutional blocks, each followed by ReLU activation and max-pooling. This is followed by a fully connected classifier.

#### Table: CNN Architecture

| **Layer**                       | **Output Size**        | **Parameters**  |
|--------------------------------|-------------------------|-----------------|
| Input                          | 128×128×3               | --              |
| Conv(32, 3×3) + ReLU           | 128×128×32              | 896             |
| MaxPool(2×2)                   | 64×64×32                | --              |
| Conv(64, 3×3) + ReLU           | 64×64×64                | 18,496          |
| MaxPool(2×2)                   | 32×32×64                | --              |
| Conv(128, 3×3) + ReLU          | 32×32×128               | 73,856          |
| MaxPool(2×2)                   | 16×16×128               | --              |
| Flatten                        | 32,768                  | --              |
| Dense(128) + ReLU              | 128                     | 4,194,432       |
| Dropout (p = 0.5)              | 128                     | --              |
| Dense(15) + Softmax            | 15                      | 1,935           |
| **Total**                      | --                      | **4,289,615**   |

**Training routine:**  
The model was trained for 10 epochs using the Adam optimizer with a learning rate of `1e-3`. We used a batch size of 32 and categorical cross-entropy loss. The training-validation split was 80/20.

**Performance:**  
The training converged quickly. As shown in the accuracy curve (see figure), the model achieved **98.68%** top-1 accuracy on a 608-image test set. The final validation loss after 9 epochs was **0.0228**.

- **Top-1 Accuracy (FP32):** 98.68%  
- **Final Loss:** 0.0228

![Accuracy curve for the baseline CNN model](cnn_acc_download.png)

**Model footprint:**  
After training, the model was exported to TensorFlow Lite (TFLite) in both full-precision (FP32) and quantized (INT8) formats.

#### Table: TFLite Model Size and Accuracy

| **Model**            | **Size** | **Accuracy** |
|----------------------|----------|--------------|
| TFLite FP32          | 16.4 MB  | 98.68%       |
| TFLite INT8 (PTQ)    | 4.2 MB   | 98.68%       |

Even though INT8 quantization reduced the model size by around **75%**, the 4.2 MB file still exceeds the **3 MB flash memory limit** of the Arduino Nicla Vision. Therefore, we explored smaller backbone models and more aggressive compression strategies in the next stages.

---

### `MobileNetV2` (α = 0.35)

The baseline CNN exceeded the 500 kB flash budget of the Nicla Vision with a 4.2 MB INT8 model. To reduce model size, we adopted **MobileNetV2** with width multiplier **α = 0.35**.

**Implementation:**
- **Input size:** `128×128`, later `96×96` for ablation.
- **Weights:** ImageNet pre-trained.
- **Head:** `GlobalAvgPool → Dense(128) → Dropout(0.3) → Dense(15, softmax)`
- **Training:** Freeze feature extractor for 5 epochs, unfreeze last 30 layers for fine-tuning.
- **Optimizer/Loss:** Adam, categorical cross-entropy

**Resolution ablation (`96×96`):**
- **INT8 accuracy:** 76%
- **INT8 size:** 772 kB

This shows that in depthwise-separable architectures, model size is dominated by **channel counts** rather than input resolution.

**Next Steps:**  
While MobileNetV2 achieves the accuracy target, the custom model is still not small enough to deploy on Nicla Vision. Future work involves further quantization using Edge Impulse.

The INT8 model reduces the FP32 size by approximately **60%**, achieving **91.28% accuracy** with a **779 kB** model size—demonstrating significant model compression with minimal accuracy loss.


### Deployment

Initial custom-trained model (MobileNetV2 with INT8 quantization) resulted in a model size of ~700 KB, which exceeded the usable flash/RAM limits of the Nicla Vision board. Therefore, we migrated to Edge Impulse Studio, which allowed us to optimize and quantize the model further using FOMO (Faster Objects, More Objects). The final INT8 quantized model from Edge Impulse was ~57 KB, making it lightweight and suitable for real-time inference on the Nicla Vision.

## Future Work

To enhance the current system into a robust, self-sufficient product, several key improvements can be done: 
- The model will be extended to recognize a wider range of Indian Sign Language (ISL) gestures beyond the initial 15 static signs. This will involve expanding the dataset and incorporating dynamic gestures using temporal models for improved communication capabilities.
- The hardware setup will be further optimized by integrating a compact LCD screen alongside the existing camera module to provide real-time textual feedback directly on the device. The predicted gesture label will be displayed both visually through an on-screen overlay and as readable text on the LCD, making the system more intuitive for nonsigners. 
- To ensure portability and user-friendliness, the entire assembly—including the camera, microcontroller, and display—will be enclosed within a custom-designed 3D-printed case. This enclosure will not only protect the internal components but also provide an ergonomic and visually appealing finish suitable for use in public or professional environments.
- Additional enhancements may include battery-powered operation for full mobility, support for voice output via a mini speaker, and Wi-Fi or Bluetooth connectivity for integration with mobile apps or assistive systems. These future developments aim to make the sign language translator a practical tool for inclusive communication in real-world scenarios.

## References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520. [Available at this link](https://doi.org/10.1109/CVPR.2018.00474)

2. TensorFlow Lite Model Optimization Toolkit. [Available at this link](https://www.tensorflow.org/lite/performance/model_optimization)

3. OpenMV IDE and Machine Vision Library. [Available at this link](https://openmv.io/pages/download)

4. Arduino Nicla Vision Documentation. [Available at this link](https://docs.arduino.cc/hardware/nicla-vision)
