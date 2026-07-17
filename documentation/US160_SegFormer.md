### User Story 160 - Comprehensive Benchmark & Environment Setup Report
**1. Environment Initialization & Verification**
- **Frameworks:** Successfully configured PyTorch and HuggingFace Transformers under `training/SegFormer/`.
- **Downstream Task Integration:** Validated model architecture with `num_labels=4` to accommodate Lumiere's semantic segmentation targets (Classes: Background, Vitiligo, Melasma, and Wine Stain).

**2. Cross-Hardware Benchmark Performance**
The following evaluations simulate 50 forward passes per model on a single face image at a standard web-resolution of $512 \times 512$ pixels.

#### Table A: CPU Baseline Performance (No Hardware Acceleration)
| Model Variant | Parameters (M) | Average Latency (ms) | Web API UX Assessment |
| :--- | :---: | :---: | :--- |
| **SegFormer-B0** | 3.72 M | 83.55 ms | 🟢 Acceptable response (< 100 ms) |
| **SegFormer-B1** | 13.68 M | 118.59 ms | 🟡 Minor delay (~ 120 ms) |
| **SegFormer-B2** | 27.35 M | 403.71 ms | ❌ High latency (Unusable for real-time) |
| **SegFormer-B3** | 47.23 M | 462.20 ms | ❌ Heavy lagging (~ 0.5 sec) |
| **SegFormer-B4** | 64.00 M | 618.15 ms | ❌ Excessive delay (~ 0.6 sec) |
| **SegFormer-B5** | 84.60 M | 707.73 ms | 🚨 Critical bottleneck (Blocks pipeline) |

#### Table B: GPU Accelerated Performance (NVIDIA GeForce RTX 3060 Ti)
| Model Variant | Parameters (M) | Average Latency (ms) | Speedup vs CPU | GPU VRAM Usage |
| :--- | :---: | :---: | :---: | :---: |
| **SegFormer-B0** | 3.72 M | 6.79 ms | 12.3x | 25.58 MB |
| **SegFormer-B1** | 13.68 M | 10.05 ms | 11.8x | 63.63 MB |
| **SegFormer-B2** | 27.35 M | 23.77 ms | 17.0x | 115.80 MB |
| **SegFormer-B3** | 47.23 M | 32.21 ms | 14.3x | 191.66 MB |
| **SegFormer-B4** | 64.00 M | 42.05 ms | 14.7x | 255.66 MB |
| **SegFormer-B5** | 84.60 M | 50.97 ms | 13.9x | 334.66 MB |

**3. Architectural Analysis & Final Selection**
- **Selected Variant:** **SegFormer-B2**
- **Justification:**
  - **The Latency Budget Constraint:** Lumiere's back-end pipeline (`SegFormer -> BiSeNet -> Claude Vision`) runs sequentially. Under a CPU environment, any tier above B1 introduces a massive latency bottleneck (> 400 ms), forcing a compromise on model accuracy.
  - **The Power of Hardware Acceleration:** Enabling the RTX 3060 Ti unlocks an average of **12x to 17x speedups**. This paradigm shift makes previously computationally restrictive models highly viable.
  - **The Golden Ratio (Precision vs Speed):** **SegFormer-B2** executes in just **23.77 ms** on the GPU, effortlessly clearing our real-time constraint (sub-50 ms threshold). Compared to B0, B2 offers a 7.3x increase in total parameters (27.35M vs 3.72M), substantially empowering the backbone network to model complex, pixel-level semantic boundaries for facial vitiligo and melasma patches.
  - **Memory Efficiency:** Active GPU VRAM allocation is exceptionally lightweight at 115.80 MB, ensuring full system stability under concurrent web request loads.

### 4. Melasma Independent-Model Improvement (2026-07-16)

The deployed melasma branch now uses a binary disease-vs-rest SegFormer trained with
cross-disease negatives from the vitiligo and port-wine-stain datasets. Training used
an NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.11.0 with CUDA 12.6 runtime, 512 x 512
inputs, batch size 2, lesion-focused crops, and low-contrast augmentation.

Checkpoint promotion requires all of the following validation gates:

| Metric | Minimum | Selected checkpoint |
| :--- | ---: | ---: |
| Pixel IoU | 0.40 | **0.5627** |
| Pixel precision | 0.60 | **0.7517** |
| Pixel recall | 0.50 | **0.6912** |
| Positive-image recall | 0.80 | **0.8125** |
| Negative-image false-positive rate | monitored | **0.0805** |

The selected checkpoint was produced at epoch 24. Independent test evaluation selected
threshold `0.50`, with IoU `0.4615`, precision `0.6719`, and recall `0.5957`. At this
threshold, the vitiligo and port-wine-stain test sets produced no cross-disease melasma
pixels. Validation gains were larger than test gains, so the system also retains a
conservative multimodal fallback for diffuse low-contrast cases: a soft SegFormer
candidate must be supported by spot observations in at least two matching facial zones.

The project requirements intentionally remain CUDA-neutral. CPU installations use the
standard PyTorch package; GPU deployments install an appropriate official CUDA wheel
and are selected automatically at runtime.



**3. Architectural Analysis & Final Selection**
- **Selected Variant:** **SegFormer-B2**
- **Justification:**
  - **The Latency Budget Constraint:** Lumiere's back-end pipeline executes strictly in sequence (`SegFormer Segmentation (US168) -> BiSeNet Tone Extraction (US169) -> Claude Vision Analysis (US170)`). Given that Large Language Model (LLM) reasoning has inherent hardware constraints, the web API enforces a strict **sub-50 ms "golden latency threshold"** for the initial image-processing stages to ensure a seamless 1.5 to 2-second total response time on the frontend.
  - **Server-Side GPU Paradigm Shift (Client-Server Architecture):** While a pure CPU execution environment severely bottlenecks performance with B2 taking up to 403.71 ms, our system adopts a dedicated Client-Server architecture. AI computation is entirely isolated on the back-end production server (e.g., powered by an NVIDIA RTX 3060 Ti), unlocking a massive 11x to 17x speedup via the CUDA compute platform. Consequently, end-users can seamlessly enjoy the full power of server-side hardware acceleration from any client device—including lower-end smartphones or standard office laptops—without any local hardware restrictions.
  - **Precision vs. Responsiveness:** **SegFormer-B2** as our optimal architectural choice.  Clocking in at an exceptional **23.77 ms** under GPU execution, it effortlessly clears the 50 ms latency threshold. Furthermore, it ensures robust high-concurrency capabilities, preventing server queuing or traffic congestion even when multiple users (e.g., 2 to 3 users simultaneously) upload photos at peak hours. In terms of capacity, B2 expands the total parameters by 7.3x compared to B0 (27.35M vs. 3.72M). This substantially empowers the backbone network to represent intricate, pixel-level semantic boundaries for facial vitiligo and melasma patches, all while maintaining an incredibly lightweight footprint of just 115.80 MB of VRAM.



### User Story 160 - 基準測試與環境建置完整報告（雙表格呈現）

1. 環境建置與驗證說明
- **開發環境：** 已成功在 `training/SegFormer/` 資料夾下配置完畢 PyTorch 與 HuggingFace Transformers 環境。
- **標籤與相容性驗證：** 完成客製化下游任務（Downstream Task）配置，設定 `num_labels=4`（分割類別：背景、白斑症、黃褐斑、鮮紅斑痣）。

2. 跨硬體基準測試數據
本測試在標準網頁影像解析度 $512 \times 512$ 像素下，針對 B0 至 B5 變體進行基準測試，每個模型均連續執行 50 次前向傳播（Forward Pass）以取得精確的平均延遲。

#### 表格 A：CPU 基準效能數據（無硬體加速）
| 模型版本 | 參數大小 (M) | 平均推論時間 (ms) | 網頁前端 UI 體驗評估 |
| :--- | :---: | :---: | :--- |
| **SegFormer-B0** | 3.72 M | 83.55 ms | 🟢 回應迅速（小於 100 ms） |
| **SegFormer-B1** | 13.68 M | 118.59 ms | 🟡 輕微延遲（符合基本要求） |
| **SegFormer-B2** | 27.35 M | 403.71 ms | ❌ 卡頓明顯（不適合實時 API） |
| **SegFormer-B3** | 47.23 M | 462.20 ms | ❌ 嚴重延遲（耗時將近 0.5 秒） |
| **SegFormer-B4** | 64.00 M | 618.15 ms | ❌ 體驗極差（耗時超過 0.6 秒） |
| **SegFormer-B5** | 84.60 M | 707.73 ms | 🚨 系統瓶頸（會造成後端管線卡死） |

#### 表格 B：GPU 硬體加速效能數據（NVIDIA GeForce RTX 3060 Ti）
| 模型版本 | 參數大小 (M) | 平均推論時間 (ms) | 速度提升倍數 (vs CPU) | GPU 顯存佔用 (VRAM) |
| :--- | :---: | :---: | :---: | :---: |
| **SegFormer-B0** | 3.72 M | 6.79 ms | 🚀 提升 ~12.3 倍 | 25.58 MB |
| **SegFormer-B1** | 13.68 M | 10.05 ms | 🚀 提升 ~11.8 倍 | 63.63 MB |
| **SegFormer-B2** | 27.35 M | 23.77 ms | 🚀 提升 ~17.0 倍 | 115.80 MB |
| **SegFormer-B3** | 47.23 M | 32.21 ms | 🚀 提升 ~14.3 倍 | 191.66 MB |
| **SegFormer-B4** | 64.00 M | 42.05 ms | 🚀 提升 ~14.7 倍 | 255.66 MB |
| **SegFormer-B5** | 84.60 M | 50.97 ms | 🚀 提升 ~13.9 倍 | 334.66 MB |

*註：顯存欄位為 PyTorch 在 GPU 執行時分配的實時硬體記憶體。在 CPU 模式下，模型直接調用系統主記憶體（RAM），故系統追蹤之 CUDA 顯存會常規顯示為 0.00 MB。*

3. 技術決策與評估結論
- **最終選定模型：SegFormer-B2**
- **核心架構決策理由：**
  - **串聯型 API 的時間預算限制：** Lumiere 後端管線採序列式串聯運行（`SegFormer 疾病遮罩(US168) -> BiSeNet 膚色提取(US169) -> Claude 大模型分析(US170)`）。由於大模型生成文字有其硬體極限，為了確保網頁端 1.5 ~ 2 秒內返回結果的流暢體驗，前端影像處理與疾病辨識步驟被分配了極其嚴苛的「50 毫秒黃金時間預算（sub-50 ms threshold）」。
  - **伺服器端硬體加速（主從式架構）：** 在純 CPU 執行環境下，B2 耗時高達 403.71ms，然而，本系統採用主從式架構（Client-Server），AI 運算完全抽離並鎖定在後端伺服器（EX:NVIDIA RTX 3060 Ti），經 CUDA 平台加速後帶來 11 至 17 倍的效能飛躍。這意味著全世界的使用者不論使用何種低配手機或文書筆電開啟網頁，都能享受伺服器帶來的超高速推論。
  - **精準度與響應速度：** **SegFormer-B2** 是我們最完美的架構解。在 GPU 上它僅需 **23.77 毫秒** 即可完成全臉疾病分割，完美守住 50 毫秒防線，並且在實際網頁上線、面對多用戶同時請求的場景(如果有 2、3 個使用者「同時」上傳照片)時也不會造成伺服器排隊塞車。在參數容量上，B2 是 B0 的 **7.3 倍**（27.35M vs 3.72M），這極大地增強了神經網路對皮膚病變（如白斑症、黃褐斑）複雜像素邊緣的勾勒能力。此外，其VRAM僅需 115.80 MB。
