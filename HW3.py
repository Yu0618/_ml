import numpy as np

# 1. 激活函數 (Sigmoid) 及其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. 準備訓練數據 (XOR 邏輯門)
# 輸入: [A, B]
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# 預期的正確輸出
y = np.array([[0], [1], [1], [0]])

# 設定隨機種子，確保每次執行結果相同
np.random.seed(42)

# 3. 初始化權重 (Weights) 與偏置 (Biases)
# 輸入層(2個節點) -> 隱藏層(3個節點) -> 輸出層(1個節點)
input_size = 2
hidden_size = 3
output_size = 1

# 隨機初始化權重
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# 初始化偏置為零
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))

# 4. 超參數設定
learning_rate = 0.5  # 學習率
epochs = 10000       # 迭代步數

print("開始訓練神經網路...")
print("-" * 30)

# 5. 訓練迴圈
for epoch in range(epochs):
    # --- 前向傳播 (Forward Propagation) ---
    # 輸入層 -> 隱藏層
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # 隱藏層 -> 輸出層
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # --- 計算誤差 (Loss) ---
    error = y - predicted_output
    
    # 每 2000 次列印一次目前的平均絕對誤差
    if epoch % 2000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch:5d} - 誤差 (Loss): {loss:.4f}")
        
    # --- 反向傳播 (Backward Propagation) ---
    # 計算輸出層的梯度 (Gradient)
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # 計算隱藏層的誤差與梯度
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # --- 更新權重與偏置 (Gradient Descent) ---
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_input_hidden = bias_hidden + np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("-" * 30)
print("訓練完成！")

# 6. 測試訓練結果
print("\n測試預測結果：")
for i in range(len(X)):
    print(f"輸入: {X[i]} -> 預測值: {predicted_output[i][0]:.4f} (預期: {y[i][0]})")
