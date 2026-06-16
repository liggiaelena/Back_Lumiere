# 🎨 Lumiere - 智能皮肤分析系统

> **一个使用 AI 技术帮助用户分析面部皮肤状况、色调和最佳妆容产品推荐的智能系统**

---

## 📖 文档导航

### 🆕 **第一次看这个项目？从这里开始**

| 文档 | 说明 | 适合 |
|------|------|------|
| [QUICKSTART_ZH.md](./QUICKSTART_ZH.md) | 📚 **完整新人上手指南** | 所有新成员必读 |
| [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md) | 🏗️ **项目架构和技术栈** | 想深入理解系统设计 |
| [DEVELOPER_FAQ.md](./DEVELOPER_FAQ.md) | ❓ **常见问题 + 快速参考** | 开发过程中遇到问题时查看 |

### 📄 **项目内技术文档**

| 文档 | 说明 |
|------|------|
| [Back_Lumiere/README.md](./Back_Lumiere/README.md) | 后端技术栈和项目结构 |
| [Front_Lumiere/README.md](./Front_Lumiere/README.md) | 前端功能和设置 |
| [Back_Lumiere/documentation/US186_API Description.md](./Back_Lumiere/documentation/US186_API%20Description.md) | 完整 API 规范 |
| [Back_Lumiere/documentation/US160_SegFormer.md](./Back_Lumiere/documentation/US160_SegFormer.md) | 模型技术详解 |

---

## 🚀 5 分钟快速开始

### 最少配置（后端 + 前端）

```bash
# Terminal 1: 启动后端 API
cd Back_Lumiere
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
# 编辑 .env，添加 ANTHROPIC_API_KEY=sk-ant-xxx
python dev/run.py

# Terminal 2: 启动前端
cd Front_Lumiere
npm install
npm run dev
```

✅ 打开浏览器 → http://localhost:5173 → 上传照片测试

---

## 💡 项目概览

### 核心功能
- ✅ 面部照片**上传/拍摄**
- ✅ AI 智能**区域分析**（额头、脸颊、鼻子、下巴）
- ✅ **皮肤色调识别**（RGB 值、Fitzpatrick 色型、底色）
- ✅ **瑕疵检测**（毛孔、痘痘、泛红、日晒斑、油光）
- ✅ **妆容推荐**（粉底液、口红、腮红等）
- ✅ **多语言支持**（英文、葡萄牙文、法文、简体中文、繁体中文、土耳其文）

### 技术亮点
| 组件 | 技术 | 用途 |
|------|------|------|
| 🖼️ 面部检测 | MediaPipe | 精确检测面部 468 个特征点 |
| 🎨 色调识别 | BiSeNet | 提取真实皮肤 RGB 值 |
| 🤖 智能分析 | Claude Vision | 区域级深度皮肤评估 |
| 💄 推荐引擎 | 自定义算法 | 匹配最佳妆容产品 |
| 🌐 前端界面 | React + Vite | 现代化响应式 UI |

---

## 📊 项目架构一览

```
用户浏览器
    ↓
前端 (React + Vite) - localhost:5173
    ↓ HTTP API
后端 (FastAPI) - localhost:8001
    ├─ MediaPipe (面部检测)
    ├─ Claude Vision (区域分析)
    ├─ BiSeNet (皮肤提取)
    └─ 推荐引擎
    ↓
SQLite 数据库 (结果持久化)
```

详细架构请查看 [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md)

---

## 📁 项目结构速览

```
Lumiere/
├── 📚 文档 (新人必读)
│   ├── QUICKSTART_ZH.md          ← 完整上手指南
│   ├── PROJECT_ARCHITECTURE.md   ← 系统架构详解
│   └── DEVELOPER_FAQ.md          ← 常见问题速查
│
├── Back_Lumiere/                 # Python 后端
│   ├── dev/app/
│   │   ├── main.py              # FastAPI 路由
│   │   ├── pipeline.py          # 分析流程编排
│   │   ├── vision.py            # Claude Vision 集成
│   │   ├── mediapipe_utils.py   # 人脸检测
│   │   ├── skin_tone_analyzer.py# 皮肤提取 (BiSeNet)
│   │   ├── recommendations.py   # 妆容推荐
│   │   └── ...其他模块
│   ├── training/                # 模型训练
│   ├── data-collection/         # 数据集
│   ├── documentation/           # 技术文档
│   └── requirements.txt          # Python 依赖
│
└── Front_Lumiere/              # React 前端
    ├── src/
    │   ├── components/         # React 组件
    │   ├── hooks/              # 自定义 Hook
    │   ├── services/           # API 客户端
    │   ├── i18n/               # 多语言支持
    │   ├── App.jsx             # 主应用
    │   └── main.jsx            # 入口
    ├── package.json
    └── vite.config.js
```

---

## 🎯 新人学习路径

### 阶段 1: 环境搭建（30 分钟）
1. ✅ 按照 [QUICKSTART_ZH.md](./QUICKSTART_ZH.md) 完成安装
2. ✅ 验证后端 `http://localhost:8001/docs` 可访问
3. ✅ 验证前端 `http://localhost:5173` 可打开
4. ✅ 上传一张照片测试完整流程

**成功标志**: 能看到分析结果

---

### 阶段 2: 代码浏览（1-2 小时）
1. 📖 阅读 [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md) 了解架构
2. 🔍 打开 IDE，浏览以下核心文件：
   - `Back_Lumiere/dev/app/main.py` - API 路由
   - `Back_Lumiere/dev/app/pipeline.py` - 分析流程
   - `Front_Lumiere/src/App.jsx` - UI 流程
3. 📝 在 [DEVELOPER_FAQ.md](./DEVELOPER_FAQ.md) 中查找相关概念

**成功标志**: 能解释完整的数据流（图片上传 → API 调用 → 结果显示）

---

### 阶段 3: 小任务实践（2-4 小时）
选择一个简单任务：
- ✏️ 添加新的妆容品牌推荐
- 🌐 添加新的语言翻译
- 📊 修改分析提示词（改变 Claude Vision 分析内容）
- 🎨 调整前端 UI（改颜色、布局）

参考 [DEVELOPER_FAQ.md](./DEVELOPER_FAQ.md#workflow-1-adding-support-for-new-makeup-brand)

**成功标志**: 能提交一个完整的功能改动

---

### 阶段 4: 深度学习（持续）
- 🤖 学习 MediaPipe 和 BiSeNet 工作原理
- 📚 查看 [US160_SegFormer.md](./Back_Lumiere/documentation/US160_SegFormer.md) 了解模型
- 🔬 尝试模型微调或重训练
- 🚀 参与核心特性开发

---

## 🔧 常见开发场景

### 场景 1: "我想改变 AI 分析什么"
**编辑**: `Back_Lumiere/dev/app/vision.py`

改变 Claude Vision 的分析提示词，就能改变分析内容。

[详细步骤](./DEVELOPER_FAQ.md#q-how-do-i-change-what-claude-vision-analyzes)

---

### 场景 2: "我想添加新的品牌推荐"
**编辑**: `Back_Lumiere/dev/app/recommendations.py`

添加品牌和色号，即可推荐新产品。

[详细步骤](./DEVELOPER_FAQ.md#workflow-1-adding-support-for-new-makeup-brand)

---

### 场景 3: "我想改进皮肤色调检测"
**编辑**: `Back_Lumiere/dev/app/skin_tone_analyzer.py` 或重训练 BiSeNet 模型

[模型细节](./Back_Lumiere/documentation/US160_SegFormer.md)

---

### 场景 4: "我想修改前端 UI"
**编辑**: `Front_Lumiere/src/components/` 中的相关组件

添加样式、改进交互、优化布局等。

---

### 场景 5: "分析太慢，我想优化性能"
[参考 FAQ](./DEVELOPER_FAQ.md#q-how-to-make-analysis-faster)

可以启用 GPU、缓存结果、压缩图像等。

---

## 🐛 问题排查

### 遇到问题？
1. **第一步**: 查看 [DEVELOPER_FAQ.md](./DEVELOPER_FAQ.md) 的相应部分
2. **第二步**: 按照排查清单 Debug
3. **第三步**: 检查日志和错误信息
4. **第四步**: 查看相关代码注释

### 常见问题速查表

| 问题 | 查看 |
|------|------|
| 后端启动失败 | [FAQ → Setup Issues](./DEVELOPER_FAQ.md#setup-issues) |
| 前端无法连接后端 | [FAQ → Frontend Connection](./DEVELOPER_FAQ.md#q-frontend-cant-connect-to-backend-api-error) |
| API 返回错误 | [API 文档](./Back_Lumiere/documentation/US186_API%20Description.md) |
| 分析结果不对 | [FAQ → Image Upload Issues](./DEVELOPER_FAQ.md#debugging-checklist) |
| 性能问题 | [FAQ → Performance](./DEVELOPER_FAQ.md#performance-questions) |

---

## 📱 API 速览

### 核心端点

**上传图像进行分析**
```bash
POST /api/analyze
Content-Type: multipart/form-data
file: <image.jpg>
```

**获取已保存的分析**
```bash
GET /api/analyze/{analyze_id}
```

**完整 API 文档**: 访问 `http://localhost:8001/docs` 或查看 [US186_API Description.md](./Back_Lumiere/documentation/US186_API%20Description.md)

---

## 🚀 快速命令参考

```bash
# 启动后端
cd Back_Lumiere
venv\Scripts\Activate.ps1
python dev/run.py

# 启动前端
cd Front_Lumiere
npm run dev

# 运行测试
curl http://localhost:8001/
curl -X POST -F "file=@test.jpg" http://localhost:8001/api/analyze

# 生产构建
cd Front_Lumiere && npm run build
cd Back_Lumiere && gunicorn -w 4 app.main:app
```

更多命令见 [DEVELOPER_FAQ.md → Quick Reference](./DEVELOPER_FAQ.md#-quick-command-reference)

---

## 🏆 最佳实践

1. ✅ **开发时**: 两个终端（后端 + 前端），使用 VS Code/PyCharm
2. ✅ **测试时**: 使用 `/docs` Swagger UI 或 Postman
3. ✅ **调试时**: 使用浏览器 DevTools (F12) 和后端日志
4. ✅ **提交前**: 测试完整流程 (上传 → 分析 → 结果)
5. ✅ **提交时**: 清楚的 commit 信息和相关文档

---

## 📊 项目统计

| 指标 | 数值 |
|------|------|
| **后端语言** | Python 3.9+ |
| **前端框架** | React 18+ |
| **核心依赖** | 25+ (Python), 10+ (JavaScript) |
| **支持语言** | 6 种 |
| **API 端点** | 3 个 |
| **主要模块** | 10+ (后端), 8+ (前端) |
| **推荐品牌** | 3+ (可扩展) |

---

## 🤝 团队协作

### 推荐工作流
```
Feature Branch → Pull Request → Code Review → Merge to Develop → Merge to Main
```

### 代码风格
- Python: PEP 8
- JavaScript: Airbnb 规范
- Commit: 英文清晰消息
- Comments: 复杂逻辑需有说明

### 获取帮助
- 📖 查看文档
- ❓ 问项目维护者
- 🔍 搜索 issue 历史
- 💬 团队讨论

---

## 📚 推荐进阶阅读

1. **FastAPI 深入**: https://fastapi.tiangolo.com/
2. **MediaPipe 文档**: https://developers.google.com/mediapipe
3. **React 官方教程**: https://react.dev/
4. **Anthropic API**: https://docs.anthropic.com/
5. **BiSeNet 论文**: https://arxiv.org/abs/1808.02272

---

## ✨ 项目特色

### 为什么选择 Lumiere？
- 🎯 **准确的皮肤分析** - 结合 AI 视觉和机器学习
- 🌍 **全球化** - 支持 6 种语言
- 📱 **响应式设计** - 支持手机/平板
- 🚀 **高性能** - 异步并行处理
- 🔐 **隐私友好** - 支持本地部署
- 📖 **易于扩展** - 模块化设计

---

## 🎓 学习成果

完成本项目的学习，您将掌握：
- ✅ 全栈 Web 应用开发
- ✅ AI/ML 集成
- ✅ RESTful API 设计
- ✅ 前后端协同
- ✅ 数据库设计
- ✅ 部署和运维基础

---

## 📞 支持

| 需要帮助? | 联系方式 |
|-----------|---------|
| 环境问题 | 查看 [DEVELOPER_FAQ.md](./DEVELOPER_FAQ.md) |
| 代码问题 | 查看相关文档或 issue |
| 功能建议 | 创建 GitHub issue |
| 技术讨论 | 团队会议或 Slack |

---

## 📄 许可证

[MIT License](./LICENSE) - 自由使用，保持署名

---

**🎉 欢迎加入 Lumiere 开发团队！快乐编码！**

---

## 🗺️ 快速链接

- **新人快速上手**: [QUICKSTART_ZH.md](./QUICKSTART_ZH.md)
- **系统架构详解**: [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md)
- **常见问题解答**: [DEVELOPER_FAQ.md](./DEVELOPER_FAQ.md)
- **API 完整规范**: [US186_API Description.md](./Back_Lumiere/documentation/US186_API%20Description.md)
- **模型技术详解**: [US160_SegFormer.md](./Back_Lumiere/documentation/US160_SegFormer.md)

---

**最后更新**: 2026-06-15 | **版本**: 1.0.0-alpha
