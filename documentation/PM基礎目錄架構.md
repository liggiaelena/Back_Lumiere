### 1. 基礎目錄架構（以 Sprint-0 為例）
在初始迭代（Sprint 0）中，專案包含以下標準資料夾與檔案：

```
Sprint-0/
├── README.md               # 專案說明文件
├── orchestrator.ipynb      # 主控/編排 Jupyter Notebook
├── data-collection/        # 資料收集與儲存目錄
├── dev/                    # 開發與執行腳本目錄
│   └── dev-run-v0.py       # 執行腳本 (v0 版本)
├── training/               # 產出模型與訓練結果目錄
│   └── trained-model-v0.h5 # 訓練完成的模型檔 (v0 版本)
└── documentation/          # 專案詳細文件目錄
```

---

### ⚙️ 2. 各模組與檔案的作用說明

* **`README.md`**：專案的說明文件，記錄專案背景、安裝方式與執行說明。
* **`orchestrator.ipynb`（核心編排檔）**：
  * **資料讀取**：從 `data-collection` 資料夾或資料庫中讀取資料集。
  * **呼叫腳本**：呼叫並執行放在 `dev/` 資料夾內的執行腳本（如 `dev-run-v0.py`）。
  * **儲存模型與設定**：將訓練出的中介檔案、模型檔（如 `.h5`）及設定檔儲存至 `training/` 資料夾。
* **`data-collection/`**：專門用來存放原始資料集或資料處理模組。
* **`dev/`**：存放開發中的 Python 執行腳本（如模型訓練邏輯、資料前處理腳本等）。
* **`training/`**：用來保存模型訓練結果（例如 TensorFlow/Keras 格式的 `.h5` 模型檔案）。
* **`documentation/`**：存放專案相關的規格書、API 文件或開發筆記。

---

### 🔄 3. 跨 Sprint 迭代與版本演進（Sprint-0 到 Sprint-1）

圖中右側展示了當專案從 **Sprint-0** 推進到 **Sprint-1** 並進行 **Merge and branch（合併與發起新分支）** 時，目錄與檔案演進的方式：

1. **版本號同步更新**：
   * **腳本**：在 `dev/` 中可能新增或升級為新的執行腳本。
   * **模型檔**：在 `training/` 中，除了原本保留的 `trained-model-v0.h5` 外，新增了最新訓練產出的 `trained-model-v1.h5`。
2. **`orchestrator.ipynb` 的作用**：在 Sprint 轉移與合併（Merge 0.1 版本）後，主控檔案 `orchestrator.ipynb` 會更新其流程，改為呼叫最新的腳本與讀寫新的模型版本（v1）。

---

### 💡 總結架構設計優點

1. **職責分離（Separation of Concerns）**：將「主控邏輯（Notebook）」、「執行程式（dev）」、「資料庫/資料集（data-collection）」以及「產出的模型（training）」明確拆開，避免程式碼混亂。
2. **可追蹤性與可重複性（Reproducibility & Versioning）**：模型檔案與腳本明確標註版本（`v0`, `v1`），使得在不同 Sprint 衝刺期間訓練出的模型可以被清楚追溯與對比。