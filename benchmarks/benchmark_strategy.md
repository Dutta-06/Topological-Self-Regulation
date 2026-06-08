# TSR Benchmark Strategy for Top-Tier Publication

To get a paper accepted at a top-tier venue (like NeurIPS, CVPR, or ICML), you must prove that your Adaptive NN (TSR) not only works on toy datasets, but actually rivals or beats heavily engineered architectures on parameter and FLOP efficiency across multiple domains.

Here is the ultimate matrix of datasets and the exact baseline/SOTA models you need to beat to prove your claim across Computer Vision, Tabular Data, and Time Series.

## 1. Computer Vision (Currently Supported)

| Dataset | Complexity / Purpose | Standard Baselines (To prove it works) | Efficiency SOTA (To prove it's ground-breaking) | Key Metric to Target |
| :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | Easy (32x32, 10 classes). Proves basic functionality. | VGG-11/16, ResNet-18 | MobileNetV2, DenseNet-121 | Higher accuracy at `< 2M` parameters. |
| **CIFAR-100** | Medium (32x32, 100 classes). Proves dynamic growth scales to harder tasks. | ResNet-34, VGG-19 | EfficientNet-B0, MobileNetV3 | Organic channel growth surpassing ResNet-34 accuracy. |
| **Tiny ImageNet** | Hard (64x64, 200 classes). Proves it handles complex spatial features. | ResNet-50 | EfficientNet-B1, ConvNeXt-Tiny | Better Accuracy vs. FLOPs trade-off curve. |
| **ImageNet-1K** | **The Gold Standard** (224x224, 1000 classes). | ResNet-50, ResNet-101 | **EfficientNet-B4**, **Swin-T** (Transformer) | Matching ResNet-50 accuracy with 50% fewer parameters. |

## 2. Tabular / Structured Data (MLP Domain)
*Deep Learning traditionally struggles to beat gradient boosting on tabular data. If your adaptive `tsr_linear.py` can beat XGBoost by dynamically finding the perfect MLP shape, this is a standalone NeurIPS paper.*

| Dataset | Complexity / Purpose | Standard Baselines | SOTA (To prove it's ground-breaking) | Key Metric to Target |
| :--- | :--- | :--- | :--- | :--- |
| **Adult Census Income** | Easy (Classification). Good sanity check for MLPs. | Standard 3-Layer MLP, Random Forest | XGBoost, LightGBM | Surpassing a static MLP with fewer overall neurons. |
| **Higgs Boson** | Medium (Physics/Classification). 11M rows, highly non-linear. | Wide ResNet-MLP | XGBoost, TabNet | Matching TabNet accuracy while proving TSR stops growing when it has enough capacity. |
| **Rossmann Store Sales** | Hard (Regression). Highly variable real-world data. | Standard 5-Layer MLP | **FT-Transformer**, **XGBoost** | Beating XGBoost RMSE using a dynamically grown architecture. |
| **Covertype (Forest)** | Hard (Multi-class Classification). 581k rows, 54 features. | Standard MLP | TabNet, NODE (Neural Oblivious Decision Ensembles) | Higher accuracy than TabNet using pure linear layers. |

## 3. Time Series Forecasting (1D Conv / RNN Domain)
*If you expand TSR to `tsr_conv1d.py` or adaptive LSTMs, you can challenge the heavily engineered Transformers that currently dominate this field.*

| Dataset | Complexity / Purpose | Standard Baselines | SOTA (To prove it's ground-breaking) | Key Metric to Target |
| :--- | :--- | :--- | :--- | :--- |
| **M4 / M5 Competition** | Medium (Retail/Sales Forecasting). Short to medium horizon. | ARIMA, Standard LSTM | N-BEATS, DeepAR | Lower MAPE (Mean Absolute Percentage Error) than static LSTMs. |
| **ETTh1 / ETTh2** | Hard (Electricity Transformer Temp). Long sequence forecasting. | TCN (Temporal ConvNet), GRU | **Informer**, **Autoformer** | Proving a dynamically grown TCN can beat Informer's attention mechanism on long-range data. |
| **Traffic / Exchange Rate**| Hard (Multivariate forecasting). Highly erratic data. | Vector Autoregression (VAR) | **PatchTST** (Transformer SOTA) | Lower MSE than PatchTST by organically growing 1D filters to capture seasonality. |

## How to use this table for a Paper:
1. **The Core Claim:** Use CIFAR-100 (CV) and Higgs Boson (Tabular) to prove that TSR's dynamic growth algorithm mathematically outperforms static depth/width across entirely different modalities.
2. **The "Holy Grail" Claim:** If your `tsr_linear.py` can organically grow an MLP that rivals **XGBoost** or **TabNet** on tabular data, you have solved one of the biggest open problems in Deep Learning today.
3. **The Ablation:** Always include the "Static Topology" ablation (running a fixed model with the exact shape TSR ultimately discovered) to prove that the *act of growing* acts as a regularizer.
