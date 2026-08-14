# Lumière AI Hybrid Analysis Pipeline

## 1. 系統架構流程圖 (System Architecture Flow)

```mermaid
graph TD
    User([使用者上傳相片]) --> API[FastAPI: POST /api/analyze]
    API --> M1[模組 1: 影像前置處理 image_utils]
    
    %% M1 直接交給 M2 與 M3，維持非破壞性全臉影像推理
    M1 -->|StandardizedImage| M2[模組 2: 臉部幾何定位器 mediapipe_utils]
    M1 -->|StandardizedImage| M3[模組 3: 全局多疾病掃描調度器 segmentation]
    
    subgraph AI Pipeline [核心 AI 模組化與空間映射管線]
        %% M2 產出全臉幾何 ROI 遮罩
        M2 -->|5大區域幾何 ROI Masks| Map[幾何空間映射 Spatial Mapping]
        
        %% M3 進行全局並行推理，避免 Crop 造成模型失準，同時大幅降低 GPU 記憶體開銷
        subgraph M3_Internal [模組 3 內部: 全局專家並行推理]
            M3 --> ParallelScan[非同步並行調度]
            ParallelScan --> Seg1[黑斑專家模型 \n 1次全臉推理]
            ParallelScan --> Seg2[白斑專家模型 \n 1次全臉推理]
            ParallelScan --> Seg3[皺紋/其他專家模型 \n 1次全臉推理]
        end
        
        %% 空間映射與融合 (利用矩陣按位與運算：Merged Mask AND ROI Mask)
        Seg1 & Seg2 & Seg3 -->|全臉疾病 Merged Mask| Map
        Map -->|分區局部疾病 Sub-Masks| M4[模組 4: 膚色分析器 skin_tone_analyzer]
    end

    %% M4 結合 M2 的分區資訊與映射後的 Sub-Masks
    M2 -->|5大區域資訊| M4

    %% M4 內部跨區域校正
    subgraph M4_Internal [模組 4 內部: 跨區域校正邏輯]
        M4 --> Step1{該區域乾淨皮膚 < 20% ?}
        Step1 -->|No, 正常計算| Normal[計算該區中位數 Median RGB]
        Step1 -->|Yes, 觸發異常| Symmetrical[借用對側/全局健康膚色]
        Normal --> Step2[產出 Skin Color Report]
        Symmetrical -->|覆蓋該區基準| Step2
    end

    %% M5 VLM 決策
    M2 -->|Crops 影像| M5[模組 5: VLM 決策引擎 vision]
    Map -->|分區病灶面積與天然部位標籤 context| M5
    Step2 -->|健康膚色基準| M5
    
    subgraph M5_Internal [模組 5 內部: LLM 脈絡提示]
        M5 --> Prompt[注入 Bias Hint 備註: \n 'Left cheek baseline skin tone biased due to 99% Vitiligo']
        Prompt --> OpenAI[OpenAI Vision 綜合評估]
    end
    
    %% 決策與後處理 (整合舊版邏輯)
    OpenAI --> FinalReport[color_utils: 彙整綜合診斷報告]
    FinalReport --> Triage{medical_alert: 醫療警告檢查 \n melanoma_suspected >= 0.85 ?}
    Triage -->|Yes, 觸發警報| Block[強制封鎖美妝推薦 \n 清空推薦清單 + 標記紅字警告]
    Triage -->|No, 正常| RecOK[recommendations: 尋找 44 筆粉底 \n 附加遮瑕/校色技巧說明]
    
    Block --> SaveDB[data_service: 寫入 PostgreSQL JSONB]
    RecOK --> SaveDB
    
    SaveDB --> Response([回傳分析結果 JSON + id + 可視化疊加圖 condition_overlay])
```

---

### 2. 管線運作步驟詳解 (Pipeline Steps)

#### Step 1：起點 — M1（影像前置處理器）

* **實作組件**：`image_utils.py`
* **運作邏輯**：
1. 接收使用者上傳之原始照片二進位檔，進行圖檔驗證與 BGR 轉 RGB。
2. 自動偵測與校正手機拍攝時產生的 EXIF 旋轉方向，過濾過小（單邊 < 300px）、亮度過暗（平均亮度 < 20）或亮度過亮/過曝（平均亮度 > 235）的劣質相片。
3. 將影像等比例縮放至設定的最大邊長（預設為 1024px），打包好這張乾淨、標準化的照片（稱為 `StandardizedImage`），然後**交給模組 2（M2）進行人臉幾何定位**。



### Step 2：人臉幾何定位與分區遮罩 — M1 ➔ M2
* **實作組件**：`mediapipe_utils.py`
* **運作邏輯**：
  1. 模組 2 接收標準照片 `StandardizedImage`。
  2. 利用 MediaPipe Face Landmarker 定位人臉特徵點，計算出額頭（`testa`）、左頰（`bochecha_e`）、右頰（`bochecha_d`）、鼻子（`nariz`）、下巴（`queixo`）這 5 大關鍵分區的幾何凸包（Convex Hull）。
  3. **非破壞性產出**：不進行物理裁切，而是產出 5 大區域的幾何 ROI 遮罩（ROI Masks），並輸出對稱區域映射表。這能確保後續模型在全臉維度進行精準推理，同時保留分區邊界資訊。

### Step 3：全局多疾病掃描與幾何空間映射 — M2 ➔ M3
* **實作組件**：`segmentation.py`（調度與空間映射）
* **運作邏輯**：
  1. **模組 3（調度器）** 接收 `StandardizedImage` 與 M2 的幾何 ROI 遮罩。
  2. **全局並行推理（規避訓練分佈不相容風險）**：直接將「全臉大圖」輸入各個繼承自 `BaseDiseaseModel` 的專家模型進行並行推理。這確保模型自注意力機制能完全發揮作用，精準捕捉病灶，並將總推理次數控制在極低範圍（4 次全臉推理 vs 20 次區域推理），防止 GPU 顯示記憶體崩潰。
  3. **空間幾何映射（Spatial Mapping）**：
     利用 Numpy 矩陣運算，將全臉疾病分割遮罩（`Merged Mask`）與五大分區的 ROI 遮罩進行交集運算（`Bitwise AND`）：
     $$\text{Local Sub-Mask}_{i} = \text{Merged Mask} \land \text{ROI}_{i}$$
     這能以近乎零的計算開銷，天然產出帶有部位屬性標籤的分區局部遮罩（例如：左臉頰白斑），完美對接前端介面的「部位疾病標籤」呈現需求。



#### Step 4：膚色計算與極端校正 — M2 + M3 ➔ M4（健康膚色分析器）

* **實作組件**：`skin_tone_analyzer.py`
* **運作邏輯**：
* **模組 4** 負責計算各分區「正常皮膚的膚色基準值」。它結合 **模組 2** 的臉部分區幾何位置與 **模組 3** 的 `Merged Mask`，進行「去瑕疵健康膚色」中位數計算，並導入 **「跨區域校正防禦機制」**：
1. **健康比例審查**：以 BiSeNet 的皮膚區域分割（Skin Label = 1）為基底，計算該分區內排除病灶遮罩後的真實乾淨皮膚佔比：

$$\text{Clean Skin Ratio} = \frac{\text{BiSeNet Skin Pixels} - \text{Disease Mask Pixels}}{\text{BiSeNet Skin Pixels}}$$


2. **正常分支**：若 $\text{Clean Skin Ratio} \ge 20\%$，扣除病灶後，直接計算該分區剩餘乾淨皮膚像素的中位數（`Median RGB/HEX`）。
3. **異常防禦分支（如 99% 左臉白斑/大面積色沉）**：若 $\text{Clean Skin Ratio} < 20\%$，為防止基準膚色被嚴重病灶強行拉偏（例如被白斑判定為純白），系統會自動觸發警報，**跨區域借用對側（如右臉頰）或全局（如額頭）的健康基準膚色來強行覆蓋**。
4. 最終產出不受嚴重病灶干擾的膚色基準報告（`Skin Color Report`）。



#### Step 5：終點與決策 — M2 + M3 + M4 ➔ M5（VLM 評估與決策引擎）

* **實作組件**：`vision.py`、`color_utils.py`
* **運作邏輯**：
* **模組 5**（OpenAI Vision 視覺大模型）扮演總診斷醫師，收集並綜合分區影像、病灶 Context 與基準膚色。
* **動態 Prompt 注入（Bias Hint 感知）**：若某分區曾觸發模組 4 的校正機制，模組 5 會在 Prompt 中自動補上備註（例如：*「左臉頰因 99% 白斑已由系統借用右臉膚色校正，請忽略影像中的白色病灶，依基準膚色提供色差推薦」*），防止 VLM 盲從影像外觀而產生決策偏誤。
* VLM 寫出精準的分區膚質診斷後，交由 `color_utils.py` 彙整綜合報告。



#### Step 6：安全警告攔截、推薦與持久化 (Triage, Recommendations & DB)

* **實作組件**：`medical_alert.py`、`recommendations.py`、`data_service.py`
* **運作邏輯**：
1. **醫療警示檢查（Triage）**：檢查 VLM 報告中如 `melanoma_suspected`（疑似黑色素瘤）、`suspicious_lesion`（可疑皮損）或 `urgent_skin_concern` 等臨床風險指標。一旦任一指標信賴度 $\ge 0.85$，系統會**強制啟用攔截**（`recommendations_blocked = true`），**清空推薦產品清單**並產生紅色就醫警告，避免誤導患者延誤就醫。
2. **美妝配對（Recommendations）**：若判定正常，系統會使用 `Median HEX` 色值，在內建 44 筆粉底液資料庫中進行色差（Luma-Weighted Euclidean Distance $\Delta E$）配對推薦。
3. **寫入資料庫（PostgreSQL）**：生成隨機 UUID `ana_[8碼十六進制]`，將分析結果序列化並以 `JSONB` 格式儲存至資料庫，回傳最終 JSON 給前端。



---

### 3. 5 大核心模組規劃與介面定義 (Module Roles & Interfaces)

#### 🧩 模組 1：影像前置處理器 (Preprocessor)

* **職責**：負責圖片二進位串流驗證、RGB 格式標準化、手機拍攝之 EXIF 轉向自動校正、極端明暗度（< 20 或 > 235）與尺寸過小之相片攔截過濾。
* **輸出介面**：
* `img_array`: 調整大小後的 RGB 影像矩陣 (`np.ndarray`)
* `b64_str`: JPEG 格式之 Base64 編碼字串 (`str`)
* `shape`: 影像維度元組 (`tuple` [height, width])



#### 🧩 模組 2：人臉幾何與區域定位器 (Locator)

* **職責**：利用 MediaPipe 臉部網格定位 478 個特徵點，計算出五官分區的凸包遮罩，進行邊界向外 15px 擴充裁剪並去除黑髮/背景。輸出需包含對稱區域映射表（`left_cheek` $\leftrightarrow$ `right_cheek`）。
* **輸出介面**：
* `crops`: 五大分區裁剪圖像之 Base64 與矩陣字典。
* `face_geometry`: 特徵點座標與分區物理邊界資料。



#### 🧩 模組 3：多疾病分割調度與空間映射引擎 (Segmentation Dispatcher & Spatial Mapper)

* **職責**：管理多個繼承自 `BaseDiseaseModel` 的專家模型分支（如黑斑、皺紋、白斑等）。直接對全臉影像進行全局並行推理以保留幾何特徵，再利用矩陣與運算（Bitwise AND）將全局疾病遮罩與五大分區之幾何 ROI 遮罩求交集，天然產出具備部位標籤的分區局部病灶遮罩。最後繪製統一的半透明渲染與白色細框全臉疊加圖。
* **輸出介面**：
* `condition_mask`: 融合後的全臉二維多標籤分類遮罩矩陣。
* `local_sub_masks`: 按五大分區歸類的局部疾病遮罩與統計資料（`Merged Mask AND ROI Mask`）。
* `condition_overlay`: 半透明色彩疊加搭配白色細框的 Base64 全臉疊加圖檔。
* `condition_map`: 結構化疾病檢測統計（病灶佔比、分區天然部位標籤）。



#### 🧩 模組 4：健康膚色分析器 (Color Estimator)

* **職責**：以 BiSeNet 解析人臉皮膚區域為骨幹，排除模組 3 產出的各分區病灶像素。實作 **「20% 乾淨皮膚安全閥」**，當乾淨皮膚比例低於臨界值時，自動執行跨區域（對側臉頰/額頭）健康基準膚色覆蓋邏輯。
* **輸出介面**：
* `median_hex`: 代表性健康膚色 HEX 色值。
* `zones`: 包含五大分區個別經過校正防禦後的代表性 HEX 與 RGB 字典（`SkinColorReport`）。



#### 🧩 模組 5：多模態視覺決策與推薦器 (VLM Decision Engine)

* **職責**：整合分區裁剪影像與病灶統計，若遭遇嚴重病灶覆蓋，自動於 Prompt 中注入 **Bias Hint（病灶偏誤提示）**。將資料送至 OpenAI Vision，強制大模型避開病灶色彩干擾，做出 Fitzpatrick 膚色級數、副色調、油脂度與 `melanoma_suspected`（疑似黑色素瘤）等高精準度診斷與醫療風險指標。
* **輸出介面**：
* `vlm_report`: 結構化 VLM 分析報告（JSON 格式），含瑕疵度、色澤均勻度、 Fitzpatrick 等級及醫療風險信賴分數。


### 4. 開發實作細節建議

1. **矩陣與運算 (Bitwise AND) 的維度一致性**：
   MediaPipe 的 ROI Masks 大小必須與 SegFormer 輸出的遮罩大小一致（或在與運算前，使用 `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` 保持維度相同）。
2. **對稱映射表的 Fallback 處理**：
   左臉頰與右臉頰有天然的對稱關係（`bochecha_e` $\leftrightarrow$ `bochecha_d`）。而額頭（`testa`）、鼻子（`nariz`）、下巴（`queixo`）沒有自然對稱區域。若這些單一區域的 Clean Skin Ratio $< 20\%$，建議 Fallback 至「其餘所有健康區域的全局中位數」作為基準膚色覆蓋。
3. **動態 Prompt 注入語系對齊**：
   Bias Hint 拼接進 VLM Prompt 時，應與調用引數中的語系 `lang` 保持對齊。例如當 `lang="tw"` 時，Bias Hint 備註應自動翻譯為繁體中文（如：「*左臉頰因大面積白斑，已由系統借用右臉健康膚色校正，評估膚色與推薦時請予以忽略。*」），以確保大模型能完全理解提示。