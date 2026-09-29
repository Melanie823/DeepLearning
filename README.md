# DeepLearning

這是深度學習課程所做的實驗專案。

本倉庫收錄「深度學習」課程（指導老師：林紋正）的三份專案，從傳統機器學習模型出發，逐步進展到多層感知器（MLP），最後以卷積神經網路（CNN）處理醫學影像。三份專案都以醫療資料為主題，並遵循相同的研究流程：

> 任務定義 → 資料說明 → 探索性資料分析（EDA）→ 資料前處理 → 實驗設計 → 結果分析 → 結論與討論

| 專案 | 主題 | 資料型態 | 任務 | 主要方法 | 最佳結果 |
|---|---|---|---|---|---|
| [期初專案](期初專案/) | 心血管疾病預測 | 表格資料（約 7 萬筆） | 二元分類 | KNN、Decision Tree、Random Forest、Naive Bayes、SVM | SVM Accuracy 0.732／Random Forest AUC 0.791 |
| [期中專案](期中專案/) | 胎兒健康狀態分類 | 表格資料（2,126 筆） | 三類別分類 | 12 組 MLP 與 XGBoost 對照 | XGBoost Accuracy 0.931／最佳 MLP 0.910 |
| [期末專案](期末專案/) | 眼底影像疾病分類 | 影像（2,394 張） | 四類別分類 | ResNet18 遷移學習，12 組超參數實驗 | Exp11 Val Accuracy 0.975 |

## 專案簡介

### 期初專案：心血管疾病分析（Cardiovascular Data Analysis）

以健康檢查紀錄（年齡、血壓、膽固醇、血糖、生活習慣等）預測個體是否患有心血管疾病。清理異常血壓、身高體重等離群值後，比較 5 種機器學習模型在 4 種前處理策略（正規化／標準化 × 是否 One-Hot）與 5-Fold／10-Fold 交叉驗證下的表現，並以 GridSearchCV 調整超參數。

➡️ [詳細說明](期初專案/README.md)

### 期中專案：胎兒健康狀態分類（Fetal Health Classification）

以胎心監護（CTG）萃取的 21 項特徵，將胎兒健康狀態分為 Normal／Suspect／Pathological。設計 12 組 MLP 實驗（縮放方式 × 編碼方式 × 模型深度），並以 XGBoost 作為對照組，探討深度學習與樹模型在表格資料上的差異。

➡️ [詳細說明](期中專案/README.md)

### 期末專案：黃斑部病變疾病分類（Macular Degeneration Disease Classification）

以 ImageNet 預訓練的 ResNet18 對眼底影像進行四類別分類（AMD、白內障、糖尿病視網膜病變、正常），系統性比較資料增強強度、優化器、學習率與 Weight Decay 共 12 組設定。

➡️ [詳細說明](期末專案/README.md)

## 倉庫結構

```
DeepLearning/
├── README.md
├── 期初專案/     # 心血管疾病預測：5 種機器學習模型比較
├── 期中專案/     # 胎兒健康分類：MLP vs XGBoost
└── 期末專案/     # 眼底影像分類：ResNet18 遷移學習
```

每個專案資料夾內都有：

- `README.md`：專案說明
- 程式碼（Jupyter Notebook 或 Colab 連結）
- 完整報告（`.docx`／`.pdf`）
- EDA 與實驗結果圖表

## 開發環境

| 項目 | 規格 |
|---|---|
| CPU | Intel Core i7-12700H（14 核心 20 執行緒） |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU（期末專案） |
| RAM | 32 GB |
| Python | 3.12.11 |
| 主要套件 | pandas、NumPy、scikit-learn、matplotlib、seaborn、TensorFlow/Keras、XGBoost、PyTorch、torchvision |

## 作者

洪翊慈
