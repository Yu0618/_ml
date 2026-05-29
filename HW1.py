import random
import copy
import math

class Solution:
    """解的抽象基底類別"""
    def height(self):
        """計算目前解的高度（適應度），越高越好"""
        raise NotImplementedError

    def neighbor(self):
        """隨機產生一個鄰居解"""
        raise NotImplementedError


def hillClimbing(init_solution, max_fails=10000):
    """
    爬山演算法主程式
    init_solution: 初始解物件
    max_fails: 連續幾次找不到更好的鄰居就停止（避免無限迴圈）
    """
    current = init_solution
    fails = 0
    
    while fails < max_fails:
        # 1. 隨機尋找一個鄰居
        neighbor = current.neighbor()
        
        # 2. 如果鄰居的高度比現在高，就移過去
        if neighbor.height() > current.height():
            current = neighbor
            fails = 0  # 重設失敗計數
        else:
            fails += 1  # 連續失敗次數加 1
            
    return current


class TspSolution(Solution):
    def __init__(self, path, distance_matrix):
        """
        path: 一個串列，代表城市的訪問順序，例如 [0, 1, 2, ..., n-1]
              注意：內部隱含最後會回到起點，即 1=>2=>3=>...=>n=>1
        distance_matrix: 二維矩陣，distance_matrix[i][j] 代表城市 i 到城市 j 的距離
        """
        self.path = path
        self.distance_matrix = distance_matrix
        self._height = None  # 快取高度，避免重複計算

    def height(self):
        """高度 = 總距離 * -1"""
        if self._height is not None:
            return self._height
        
        total_distance = 0
        n = len(self.path)
        
        # 計算 1=>2=>3=>...=>n=>1 的總距離
        for i in range(n):
            city_from = self.path[i]
            city_to = self.path[(i + 1) % n]  # (i+1)%n 確保最後一個城市會連回第一個城市
            total_distance += self.distance_matrix[city_from][city_to]
            
        self._height = -total_distance
        return self._height

    def neighbor(self):
        """
        鄰居函數：採用 2-opt 策略。
        隨機選擇兩個不相鄰的邊 (a, b) 與 (c, d)
        將路徑中 b 到 c 的這段子路徑反轉，
        路徑就會從原本的 (..., a, b, ..., c, d, ...) 變成 (..., a, c, ..., b, d, ...)
        這等同於斷開 (a,b) 和 (c,d)，重新連接成 (a,c) 和 (b,c)
        """
        n = len(self.path)
        new_path = copy.deepcopy(self.path)
        
        # 隨機選擇兩個切點 i 和 j (確保 i < j)
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        
        # 反轉 i+1 到 j 之間的路徑
        # 例如原路徑: [..., a, b, x, y, c, d, ...]，i對應a，j對應c
        # 反轉後:     [..., a, c, y, x, b, d, ...]
        new_path[i+1 : j+1] = reversed(new_path[i+1 : j+1])
        
        return TspSolution(new_path, self.distance_matrix)

    def get_total_distance(self):
        """輔助函數：取得正值的總距離"""
        return -self.height()


if __name__ == "__main__":
    # 1. 建立測試資料：隨機產生 10 個城市在二維平面的座標
    random.seed(42)  # 固定隨機種子以利測試
    num_cities = 10
    coordinates = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_cities)]
    
    # 建立城市間的距離矩陣 (歐幾里得距離)
    dist_matrix = [[0.0] * num_cities for _ in range(num_cities)]
    for i in range(num_cities):
        for j in range(num_cities):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            dist_matrix[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
    # 2. 建立初始解：1 => 2 => 3 => ... => n => 1
    # 這裡用索引表示：[0, 1, 2, ..., num_cities-1]
    initial_path = list(range(num_cities))
    init_sol = TspSolution(initial_path, dist_matrix)
    
    print("=== 初始解 ===")
    print(f"路徑: {' => '.join(map(str, init_sol.path))} => {init_sol.path[0]}")
    print(f"總距離: {init_sol.get_total_distance():.2f}")
    
    # 3. 執行爬山演算法
    best_sol = hillClimbing(init_sol, max_fails=20000)
    
    print("\n=== 爬山演算法最佳化後的解 ===")
    print(f"路徑: {' => '.join(map(str, best_sol.path))} => {best_sol.path[0]}")
    print(f"總距離: {best_sol.get_total_distance():.2f}")
