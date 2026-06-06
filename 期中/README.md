# 🌌 AI 羅塞塔石碑 (Project Rosetta-AI)
### 基於評測工程與提示詞對齊的跨物種意圖轉譯系統

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Category](https://img.shields.io/badge/AI-Harness__Engineering-orange.svg)

本專案是一個探索大語言模型（LLM）在非文字、跨物種溝通（如動物行為、嬰兒哭聲、未知地外訊號）中解讀潛力的微型評測框架（Harness Engineering）。系統將環境脈絡、聲音頻率及肢體動作轉化為結構化特徵，並透過自動化基準測試，定量評估模型在未知語言解讀上的精準度。

---

## 📌 研究動機與背景

1. **跨物種溝通痛點**：人類與動物、新生兒或地外訊號交流時，受限於「無文字」障礙。然而這些主體仍透過聲音頻率與肢體動作傳遞強烈意圖。
2. **LLM 的高維識別**：大語言模型本質上是模式識別機器（Pattern Recognition Engine），只要給予足夠的環境脈絡（Context），就能識別非人類訊號背後的情感與動機。
3. **評測工程（Harness）核心**：在開發前沿 AI 系統時，需要有一套自動化基準測試來客觀評估 Prompt 的好壞，這正是本專案的實作重點。

---

## 🛠️ 系統架構與模組設計

本系統完全基於輕量化 Python 實作，包含以下四個核心模組：

* **基準數據集 (Benchmark Dataset)**：模擬 `Cat`（生物寵物）、`Baby`（人類發展）與 `Alien_Alpha`（地外訊號）三種維度的黃金標準（Ground Truth）。
* **提示詞工程 (Prompt Engineering)**：透過動態生成結構化提示詞，賦予模型專家角色，並嚴格約束輸出為 JSON 格式。
* **模型推理層 (LLM Inference Layer)**：模擬模型推理行為，在生產環境中可無縫對接 OpenAI API 或本地 Ollama 部署的模型。
* **評測主程式 (Harness Execution)**：自動遍歷測試集、解析回傳 JSON 並計算系統成功解析率（Success Rate）。

---

## 🚀 執行結果範例

* **執行解果.txt** : https://github.com/Yu0618/_ml/blob/master/%E6%9C%9F%E4%B8%AD/%E5%9F%B7%E8%A1%8C%E7%B5%90%E6%9E%9C.txt
