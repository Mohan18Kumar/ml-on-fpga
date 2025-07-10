# 🧠 ML on FPGA — 12 Week Learning Journey 🚀

This repository documents my learning journey in integrating Machine Learning models on FPGAs using Verilog, Vitis HLS, and Python. The final goal is to build a CNN-based digit classifier running fully on FPGA.

---

## 📅 Weekly Plan

| Week | Topic                                 | Folder                  |
|------|----------------------------------------|--------------------------|
| 1    | Verilog & Digital Logic Refresher     | 01_hdl_basics/          |
| 2–3  | Python + ML Basics (MLP)              | 02_python_ml/           |
| 4    | Fixed-Point Arithmetic                | 03_fixed_point/         |
| 5–6  | MLP on FPGA (Verilog & HLS)           | 04_mlp_fpga/            |
| 7    | Image Pre-processing                  | 05_image_processing/    |
| 8–9  | CNN Model Design & Accelerator        | 06_cnn_model/           |
| 10   | FPGA Interfaces (BRAM, UART)          | 07_fpga_interface/      |
| 11   | Full Pipeline Integration             | 08_integration/         |
| 12   | Testing, Docs, GitHub Repo            | 09_docs/                |

---

## 🧰 Tools Used

- **Xilinx Vivado** for Verilog synthesis
- **Vitis HLS** for high-level synthesis from C
- **Python (NumPy, TensorFlow)** for training ML models
- **Matplotlib** for visualizations
- **ModelSim / GTKWave** for simulation

---

## 🎯 Final Goal

Implement a quantized CNN digit classifier that runs on an FPGA with real-time performance, with UART interface for image input and classification output.

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## ROAD MAP

### ✅ Week 1: Foundations of Digital Design & Verilog Review

**Topics:**

* Combinational and sequential logic
* FSMs, datapaths, ALUs
* Simulation and synthesis in Vivado

**Goals:**

* Refresh Verilog syntax and simulation basics

**Project:**

* Design and simulate a 4-bit ALU (Add, Subtract, AND, OR)

---

### ✅ Week 2: Introduction to Python & Numpy for ML

**Topics:**

* Python basics for engineering
* Numpy array operations
* Basic plotting with Matplotlib

**Goals:**

* Learn how ML models operate in code

**Project:**

* Build a simple Perceptron classifier (2D point classification)

---

### ✅ Week 3: ML Fundamentals (MLP, Activation Functions, Inference)

**Topics:**

* Forward pass in MLP
* Activation functions: ReLU, Sigmoid, Softmax
* Manual weight application

**Goals:**

* Implement a 2-layer MLP forward pass manually

**Project:**

* Build and test a 2-layer MLP for digit recognition (MNIST, software-only)

---

### ✅ Week 4: Fixed-Point Arithmetic & Quantization

**Topics:**

* Fixed-point formats (Q1.15, Q8.8, etc.)
* Quantization of weights/inputs
* Accuracy vs performance trade-offs

**Goals:**

* Convert Python model to fixed-point representation

**Project:**

* Simulate fixed-point MLP in C or Python

---

### ✅ Week 5: FPGA Project - MLP in Verilog (1 Hidden Layer)

**Topics:**

* Multiply-accumulate (MAC) unit
* Matrix-vector multiplication in hardware

**Goals:**

* Implement a basic MLP with hardcoded weights

**Project:**

* Build a fixed-weight digit classifier in Verilog using MAC units

---

### ✅ Week 6: Introduction to High-Level Synthesis (HLS)

**Topics:**

* C to RTL concepts
* Using Vitis HLS
* Pipelining and loop unrolling

**Goals:**

* Write MLP model in C and convert to HLS

**Project:**

* Reimplement the Verilog MLP from Week 5 using Vitis HLS

---

### ✅ Week 7: Basic Image Processing in Python

**Topics:**

* Image pre-processing for CNNs
* Resizing, thresholding, normalization

**Goals:**

* Prepare inputs for FPGA ML inference

**Project:**

* Image pre-processing pipeline for FPGA-ready binary input (MNIST image format)

---

### ✅ Week 8: CNN Concepts and Simplified Model Design

**Topics:**

* Convolution, Pooling, Flattening
* Feature extraction vs classification

**Goals:**

* Train a tiny CNN with 1 conv + 1 FC layer

**Project:**

* Train & quantize a tiny CNN model for 5-class digit classification (MNIST subset)

---

### ✅ Week 9: CNN Accelerator Architecture in Verilog or HLS

**Topics:**

* Designing convolution layer
* Dataflow pipelining in HLS

**Goals:**

* Implement Conv + ReLU accelerator in hardware

**Project:**

* CNN feature extractor module using HLS (or Verilog)

---

### ✅ Week 10: Memory & Interface (BRAM, UART, AXI)

**Topics:**

* Data movement and buffering
* Loading image input from memory or UART

**Goals:**

* Interface FPGA module with external input

**Project:**

* Implement BRAM interface and UART data loading for image input

---

### ✅ Week 11: Integration and Optimization

**Topics:**

* Combine CNN + FC + Softmax
* Timing optimization, resource utilization

**Goals:**

* Full model integration

**Project:**

* Complete digit classifier pipeline in hardware with UART input and classification output

---

### ✅ Week 12: Testing, Documentation, and GitHub Upload

**Topics:**

* Functional testing and debugging
* GitHub project structuring
* Writing documentation and block diagrams

**Goals:**

* Finalize and present project

**Project:**

* Create GitHub repository with code, diagrams, testbenches, and report

---

**End Goal:** You will build an end-to-end ML pipeline (from Python training to FPGA inference), capable of classifying digits or simple patterns with real-time performance on an FPGA.
