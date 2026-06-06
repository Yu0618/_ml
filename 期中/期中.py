import json

# 1. 模擬「非人類語言」與環境特徵的數據集 (以貓、嬰兒、虛擬外星人為例)
# 這是我們的測試基準 (Benchmark Dataset)
test_dataset = [
    {
        "subject": "Cat",
        "input_signals": "頻率: 450Hz, 持續時間: 1.5秒, 身體語言: 尾巴直立微彎, 眼神: 放大瞇眼, 當時環境: 主人在廚房開罐頭",
        "ground_truth": "興奮、期待、討食"
    },
    {
        "subject": "Baby",
        "input_signals": "哭聲類型: 規律間歇性, 音調: 逐漸變高, 身體動作: 猛烈踢腿, 距離上次進食: 4.5小時",
        "ground_truth": "肚子餓、尋求餵食"
    },
    {
        "subject": "Alien_Alpha",
        "input_signals": "光信號: 藍光閃爍三次, 電磁波段: 1420MHz, 伴隨引力波: 輕微震盪",
        "ground_truth": "友善問候、確認彼此坐標"
    }
]

# 2. 定義系統提示詞 (System Prompt) - 這是 Harness 工程的核心
def get_translator_prompt(subject, signals):
    return f"""你現在是宇宙最頂尖的語言學家與生物行為學家，正在主持一項名為「AI 羅塞塔石碑」的計畫。
你的任務是解讀非人類語言。請分析以下輸入的感官訊號、環境脈絡與行為特徵，推導出其背後真正的情感與含意。

【生物/主體】: {subject}
【觀測訊號】: {signals}

請嚴格按照以下 JSON 格式回覆，不要包含任何額外解釋：
{{
    "detected_emotion": "主體當前的情感狀態",
    "decoded_meaning": "翻譯成人類語言的具體含意 (15字以內)",
    "confidence_score": "置信度 (0.0 到 1.0)"
}}
"""

# 3. 模擬 LLM 解讀邏輯 (實際開發時可替換為 openai API 或本地 Ollama 呼叫)
def mock_llm_call(prompt):
    # 這裡模擬 LLM 接收到 Prompt 後的高質量回傳
    if "Cat" in prompt:
        return '{"detected_emotion": "興奮與期待", "decoded_meaning": "快點開罐頭，我餓了", "confidence_score": 0.95}'
    elif "Baby" in prompt:
        return '{"detected_emotion": "焦慮與生理不適", "decoded_meaning": "肚子很餓想喝奶", "confidence_score": 0.88}'
    else:
        return '{"detected_emotion": "和平探索", "decoded_meaning": "你好，我們沒有敵意", "confidence_score": 0.75}'

# 4. 評測工程 (Harness Engineering) 主程式
def run_rosetta_harness():
    print("=== AI 羅塞塔石碑：跨物種語言解讀評測開始 ===")
    total_tests = len(test_dataset)
    passed_evals = 0
    
    for i, data in enumerate(test_dataset):
        print(f"\n[測試案例 {i+1}] 目標: {data['subject']}")
        print(f"輸入訊號: {data['input_signals']}")
        
        # 產生 Prompt 並送入模型
        prompt = get_translator_prompt(data['subject'], data['input_signals'])
        raw_response = mock_llm_call(prompt)
        
        # 解析結果
        try:
            result = json.loads(raw_response)
            print(f"-> AI 解讀情感: {result['detected_emotion']}")
            print(f"-> AI 翻譯含意: {result['decoded_meaning']}")
            print(f"-> 模型置信度: {result['confidence_score']}")
            print(f"-> 預期真實含意: {data['ground_truth']}")
            
            # 簡易自動化評測對齊 (Harness 核心)
            # 實務上會用 Embedding 相似度比對，這裡用手動判斷或關鍵字模擬
            passed_evals += 1 
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            
    print("\n" + "="*40)
    print(f"評測結束。成功解析率: {passed_evals}/{total_tests} ({(passed_evals/total_tests)*100:.1f}%)")

if __name__ == "__main__":
    run_rosetta_harness()
