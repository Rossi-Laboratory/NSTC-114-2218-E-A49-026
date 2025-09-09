# Action Chunk Prediction

**目的**：提升機器人對多步行為的序列學習能力，改善傳統逐步預測的效率。  
**目標**：設計 **Action Chunk Prediction (ACP)** 框架，參考 **FAST** 與 **VQ Action Chunking**，建立初版架構與程式。

> ⚠️ 中間會用到 VLA 模型，**務必使用既有的 `VLA-MoE-Manipulation` 專案**（本庫不會另外實作 VLA）。
> 由 `acp/integration/vla_moe_bridge.py` 統一呼叫。

## 特色
- 時序編碼（Transformer/TCN）→ **邊界偵測（FAST-style）**
- 片段聚合（segment pooling）→ **VQ 碼本量化**（可選）
- 支援單層或 **CL3（3層級）** 分塊（atomic→micro→macro）
- 與 **VLA-MoE-Manipulation** 串接：由 VLA 產出低階連續動作，再由 ACP 進行分塊與序列重建

## 快速開始
```bash
# 1) 取得/更新 VLA 子模組
bash scripts/prepare_submodule.sh  # 需 git 可連外

# 2) 安裝依賴（建議 venv）
pip install -r requirements.txt
pip install -e .

# 3) 預處理序列（將 demonstrations 轉成可學資料）
python scripts/preprocess_sequences.py --input examples/manipulation/sequences/demo_seq_01.json     --output data/processed/train_sequences.pt

# 4) 訓練
bash scripts/train_acp.sh

# 5) 推論（輸出 chunk 邊界與索引）
bash scripts/infer_acp.sh "pick up the red block"

# 6) 視覺化
python -m acp.inference.visualize --input outputs/last_infer.json --save outputs/vis_chunks.png
```

## 專案結構
```
Action-Chunk-Prediction/
├─ configs/                 # 模型與訓練/推論組態
├─ acp/                     # 主程式套件（模型/資料/訓練/推論/工具）
├─ scripts/                 # 預處理/訓練/推論/評估/子模組
├─ docs/                    # 設計/資料格式/評估/整合說明
├─ examples/manipulation/   # Manipulation 示範資料
├─ data/                    # 預設資料夾（raw/processed/cache）
└─ third_party/VLA-MoE-Manipulation/  # ★ 子模組（請用你既有的 VLA 專案）
```

## 授權
本專案採用 **Apache License 2.0**。
