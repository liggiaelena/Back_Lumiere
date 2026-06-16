# Lumiere 项目 - 新人快速上手指南

## 📋 项目概览

**Lumiere** 是一个智能皮肤分析系统，使用 AI 技术帮助用户分析面部皮肤状况、色调和最佳妆容产品推荐。

### 核心功能
- 📸 **面部照片上传/拍摄** - 支持本地上传或实时摄像头拍照
- 🎯 **区域分析** - 智能检测额头、脸颊、鼻子、下巴等面部区域
- 🎨 **皮肤色调识别** - 利用 BiSeNet 模型提取真实皮肤 RGB 值
- 🤖 **AI 评估** - 调用 Claude Vision API 进行每个区域的皮肤评估（毛孔、痘痘、泛红等）
- 💄 **妆容推荐** - 推荐合适的粉底液、口红等产品（Fenty Beauty, MAC, Maybelline 等品牌）
- 🌍 **多语言支持** - 英文、葡萄牙文、法文、简体中文、繁体中文、土耳其文

---

## 🏗️ 项目架构

```
Lumiere/
├── Back_Lumiere/          # Python 后端 API 服务
│   ├── dev/
│   │   ├── app/           # FastAPI 应用核心
│   │   │   ├── main.py    # 路由和服务入口
│   │   │   ├── pipeline.py        # 分析流程编排
│   │   │   ├── mediapipe_utils.py # 人脸检测和区域提取
│   │   │   ├── vision.py          # Claude Vision API 集成
│   │   │   ├── color_utils.py     # 颜色计算工具
│   │   │   ├── skin_tone_analyzer.py # BiSeNet 皮肤分析
│   │   │   ├── recommendations.py # 妆容推荐引擎
│   │   │   ├── image_utils.py     # 图像预处理
│   │   │   ├── db.py      # 数据库连接
│   │   │   ├── data_service.py    # 数据持久化
│   │   │   └── config.py  # 配置管理
│   │   └── run.py         # 开发服务启动脚本
│   ├── training/          # 模型训练相关
│   │   ├── checkpoints/   # 预训练模型权重
│   │   │   ├── bisenet_best.pth  # BiSeNet 皮肤提取模型
│   │   │   └── SegFormer/        # SegFormer 疾病分割模型
│   │   ├── models/        # 模型架构定义
│   │   └── SegFormer/     # SegFormer 训练脚本
│   ├── data-collection/   # 数据收集和处理
│   │   ├── melasma/       # 黄褐斑数据集
│   │   ├── port_wine_stain/ # 葡萄酒色斑数据集
│   │   └── vitiligo/      # 白癜风数据集
│   ├── documentation/     # API 和技术文档
│   ├── requirements.txt    # Python 依赖
│   └── .env              # 环境变量（API 密钥等）
│
├── Front_Lumiere/        # React 前端应用
│   ├── src/
│   │   ├── components/   # React 组件
│   │   │   ├── UploadZone/       # 拖拽上传区域
│   │   │   ├── CameraCapture/    # 摄像头拍照
│   │   │   ├── FacePreview/      # 图片预览确认
│   │   │   ├── AnalysisResult/   # 结果展示页面
│   │   │   ├── Recommendations/  # 妆容推荐卡片
│   │   │   ├── RegionCard/       # 区域分析卡片
│   │   │   ├── ToneComparison/   # 色调对比
│   │   │   └── LoadingSpinner/   # 加载动画
│   │   ├── hooks/        # React 自定义 Hook
│   │   │   ├── useAnalysis.js    # API 调用和状态管理
│   │   │   └── useCamera.js      # 摄像头流管理
│   │   ├── i18n/         # 国际化
│   │   │   ├── LanguageContext.jsx # 语言上下文
│   │   │   └── translations.js     # 6 种语言翻译文本
│   │   ├── services/     # 服务层
│   │   │   └── api.js           # API 客户端（Axios）
│   │   ├── utils/        # 工具函数
│   │   │   └── colorUtils.js    # 颜色转换工具
│   │   ├── App.jsx       # 主应用组件
│   │   └── main.jsx      # 入口文件
│   ├── package.json      # 依赖配置
│   ├── vite.config.js    # Vite 配置
│   └── .env             # 环境变量（API 地址等）
│
└── Datasets/            # 训练数据
    ├── mst-e_data/      # 多肤色皮肤数据集
    └── ...
```

---

## 🚀 快速开始

### 前置条件
- Python 3.9+（后端）
- Node.js 18+（前端）
- Git
- API 密钥：Anthropic API（Claude Vision）或 Google Gemini API

### 1️⃣ 克隆项目
```bash
git clone <repository-url>
cd Lumiere
```

### 2️⃣ 安装和启动后端

#### 步骤 1: 创建虚拟环境
```bash
cd Back_Lumiere
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# 或 Windows CMD
venv\Scripts\activate

# 或 Linux/Mac
source venv/bin/activate
```

#### 步骤 2: 安装依赖
```bash
pip install -r requirements.txt
```

#### 步骤 3: 配置环境变量
在 `Back_Lumiere/` 目录创建 `.env` 文件：
```env
# 选择其中一个 API 提供商

# 方式 1: 使用 Anthropic Claude Vision (推荐)
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# 方式 2: 使用 Google Gemini Vision
GEMINI_API_KEY=your-gemini-key-here
```

#### 步骤 4: 初始化数据库（可选）
```bash
# 如需使用数据库存储分析结果
sqlite3 analyses.db < db/init.sql
```

#### 步骤 5: 启动 API 服务
```bash
# 从 Back_Lumiere 目录启动
python dev/run.py
```

✅ 服务应在 `http://localhost:8001` 运行
- 健康检查：`http://localhost:8001/`
- API 文档：`http://localhost:8001/docs`（Swagger UI）

### 3️⃣ 安装和启动前端

#### 步骤 1: 安装依赖
```bash
cd Front_Lumiere
npm install
```

#### 步骤 2: 配置环境变量（可选）
在 `Front_Lumiere/` 目录创建 `.env` 文件：
```env
# 后端 API 地址（默认已是 http://localhost:8001）
VITE_API_URL=http://localhost:8001
```

#### 步骤 3: 启动开发服务器
```bash
npm run dev
```

✅ 应用在 `http://localhost:5173` 运行

### 4️⃣ 验证集成
1. 打开浏览器访问 `http://localhost:5173`
2. 上传或拍摄一张面部照片
3. 等待分析完成（约 5-10 秒）
4. 查看结果

---

## 🔄 工作流程

### 用户端流程
```
上传/拍摄照片 → 预览确认 → 开始分析 → 加载动画 → 显示结果
```

### 后端分析流程

```
1. 图像验证和预处理
   ↓
2. MediaPipe 人脸检测 → 提取 4 个区域（额头、脸颊、鼻子、下巴）
   ↓
3. 并行处理：
   ├─ Claude Vision 分析每个区域（毛孔、痘痘、泛红、日晒斑等）
   └─ BiSeNet 提取全脸皮肤 RGB 值和 Fitzpatrick 色调
   ↓
4. 综合 → 生成最终报告（色调分类、基础妆容推荐）
   ↓
5. 保存到数据库 → 返回结果给前端
```

---

## 📱 API 接口

### 1. 健康检查
```
GET /
```
检查服务和数据库连接状态。

**响应:**
```json
{
  "status": "ok",
  "service": "skin-analyzer",
  "db": "ok"
}
```

### 2. 图像分析（核心接口）
```
POST /api/analyze
Content-Type: multipart/form-data
```

**请求:**
- `file`: 面部图片（JPG/PNG/WebP，最大 10MB）

**响应 (200):**
```json
{
  "id": "ana_a1b2c3d4",
  "skin_tone": {
    "fitzpatrick_scale": "Type III-IV",
    "undertone": "warm",
    "hex_value": "#C08552",
    "rgb": [192, 133, 82]
  },
  "regions": {
    "forehead": {
      "pores": "moderate",
      "acne": "none",
      "redness": "mild",
      "sun_spots": "few"
    },
    "left_cheek": {...},
    "right_cheek": {...},
    "nose": {...},
    "chin": {...}
  },
  "recommendations": {
    "foundation": [
      {
        "brand": "Fenty Beauty",
        "shade": "140W",
        "reason": "Matches undertone and depth"
      },
      ...
    ],
    "lipstick": [...],
    "blush": [...]
  }
}
```

### 3. 获取已保存分析
```
GET /api/analyze/{analyze_id}
```

**响应:**
```json
{
  "id": "ana_a1b2c3d4",
  "created_at": "2024-01-15T10:30:45",
  "skin_tone": {...},
  "regions": {...},
  "recommendations": {...}
}
```

---

## 🔑 核心模块详解

### 📦 后端模块

#### `pipeline.py` - 分析流程编排
**职责**: 协调整个分析流程

**关键函数**:
- `run_pipeline(img_rgb)` - 异步执行完整分析

**工作步骤**:
1. 预处理图像（缩放、归一化）
2. 使用 MediaPipe 检测面部
3. 并行运行 Claude Vision 和 BiSeNet
4. 汇总结果生成报告

---

#### `mediapipe_utils.py` - 人脸检测
**职责**: 使用 MediaPipe 进行人脸定位和区域提取

**关键函数**:
- `get_landmarks(img)` - 检测 468 个面部特征点
- `extract_region_crops(img, coords)` - 提取 4 个脸部区域的 Base64 编码

**区域定义**:
- forehead（额头）
- left_cheek & right_cheek（两侧脸颊）
- nose（鼻子）
- chin（下巴）

---

#### `vision.py` - Claude Vision 集成
**职责**: 调用 Claude AI 进行皮肤分析

**关键函数**:
- `analyze_region(region_name, base64_crop)` - 对单个区域进行 AI 分析

**分析内容**:
```
- pores (毛孔): none, mild, moderate, severe
- acne (痘痘): none, few, moderate, severe
- redness (泛红): none, mild, moderate, severe
- sun_spots (日晒斑): none, few, moderate, many
- shine (油光): none, mild, moderate, severe
```

---

#### `skin_tone_analyzer.py` - BiSeNet 皮肤分析
**职责**: 提取真实皮肤色调和色彩特征

**模型**: BiSeNet（预训练的语义分割模型）

**输出**:
```python
{
    "fitzpatrick_scale": "Type II-III",      # 皮肤色型
    "undertone": "warm|neutral|cool",        # 底色
    "hex_value": "#ABC123",                  # 十六进制颜色
    "rgb": [171, 193, 35],                   # RGB 值
    "hsv": [45, 82, 76]                      # HSV 值（可选）
}
```

---

#### `recommendations.py` - 妆容推荐
**职责**: 基于皮肤色调推荐合适的妆容产品

**品牌库**:
- Fenty Beauty（多色号选择）
- MAC（专业妆容）
- Maybelline（大众产品）
- 等更多品牌

**推荐逻辑**:
1. 根据 Fitzpatrick 色型和底色匹配色号
2. 考虑皮肤状况（油性、干性等）
3. 提供基础液、唇膏、腮红等多类产品推荐

---

#### `data_service.py` - 数据持久化
**职责**: 管理分析结果的存储和检索

**关键函数**:
- `save_analysis(result)` - 保存分析结果，返回 ID
- `get_analysis(analyze_id)` - 检索已保存的分析

---

### 🎨 前端模块

#### `useAnalysis.js` - API 集成和状态管理
**职责**: 处理图像上传和 API 调用

**状态**:
```javascript
{
  loading: boolean,     // 正在分析
  error: string|null,   // 错误信息
  result: object|null   // 分析结果
}
```

**关键函数**:
- `analyze(file)` - 提交图像进行分析
- `reset()` - 重置状态

---

#### `useCamera.js` - 摄像头管理
**职责**: 处理实时摄像头流和拍照

**功能**:
- 请求相机权限
- 显示实时预览
- 拍照并返回图片 Blob

---

#### 国际化 (`i18n/`)
**支持语言**:
- 英文 (EN)
- 葡萄牙文 (PT)
- 法文 (FR)
- 简体中文 (ZH)
- 繁体中文 (TW)
- 土耳其文 (TR)

**用法**:
```javascript
const { t } = useLanguage();
<p>{t('analyze.title')}</p>  // 自动翻译
```

---

## 🛠️ 开发建议

### 新功能开发流程

#### 1. 后端新功能
**添加新的分析指标**:
```
1. 在 vision.py 中添加提示词
2. 在 recommendations.py 更新推荐逻辑
3. 在 data_service.py 更新数据模式
4. 测试 API 端点
5. 更新 API 文档
```

**添加新的 API 路由**:
```python
# main.py
@app.post("/api/new-endpoint")
async def new_endpoint(request_data: RequestModel):
    # 处理逻辑
    return JSONResponse(content={...})
```

#### 2. 前端新功能
**添加新组件**:
```
1. 在 src/components 创建文件夹
2. 编写 JSX 组件
3. 导入到需要的地方
4. 添加多语言支持（translations.js）
5. 测试响应式设计
```

#### 3. 模型优化
```
1. 在 training/SegFormer 准备数据
2. 运行训练脚本
3. 保存到 checkpoints/
4. 更新 pipeline.py 加载新模型
5. 性能测试
```

---

## 📊 常见任务

### 调试分析流程
```bash
# 后端日志
python dev/run.py  # 查看 console 输出

# 前端调试
npm run dev        # 打开浏览器开发者工具（F12）
```

### 查看 API 文档
访问 `http://localhost:8001/docs`（Swagger UI）

### 测试 API
```bash
# 使用 curl 测试
curl -X POST -F "file=@test.jpg" http://localhost:8001/api/analyze

# 或使用 Postman/Thunder Client
```

### 生产部署
```bash
# 前端
npm run build      # 生成优化后的 dist/ 文件夹

# 后端
# 使用 Gunicorn 或 Uvicorn：
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

---

## 📝 常见问题

### Q: 后端启动失败，提示找不到模块
**A**: 确保：
1. 虚拟环境已激活
2. 运行了 `pip install -r requirements.txt`
3. 工作目录在 `Back_Lumiere/` 或项目根目录

### Q: 前端无法连接到后端
**A**: 检查：
1. 后端是否在 `http://localhost:8001` 运行
2. 前端 `.env` 中 `VITE_API_URL` 是否正确
3. 防火墙是否阻止了 8001 端口

### Q: Claude Vision API 返回错误
**A**: 
1. 确认 API 密钥有效且配额充足
2. 检查图像大小和格式（JPG/PNG/WebP）
3. 查看详细错误信息：添加 `logging.basicConfig(level=logging.DEBUG)`

### Q: 分析速度很慢
**A**:
1. 检查网络连接（Claude API 调用）
2. 图像尺寸太大可能增加处理时间
3. 考虑使用 GPU 加速 BiSeNet 模型

---

## 📚 更多资源

- **技术文档**: [US186_API Description.md](Back_Lumiere/documentation/US186_API Description.md)
- **模型文档**: [US160_SegFormer.md](Back_Lumiere/documentation/US160_SegFormer.md)
- **官方依赖**:
  - [MediaPipe](https://developers.google.com/mediapipe)
  - [FastAPI](https://fastapi.tiangolo.com/)
  - [React](https://react.dev/)
  - [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)

---

## 💡 团队协作建议

1. **代码规范**: 遵循 PEP 8（Python）和 Airbnb JS 规范
2. **分支管理**: 
   - `main` - 生产分支
   - `develop` - 开发分支
   - `feature/xxx` - 功能分支
3. **提交信息**: 使用清晰的英文提交信息
4. **代码审查**: 所有 PR 需要至少一人审查

---

**🎉 祝您开发愉快！如有问题，请查看技术文档或联系团队。**
