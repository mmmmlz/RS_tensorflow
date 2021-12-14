# -*- coding: utf-8 -*-
# @Author   ：mmmmlz
# @Time   ：2021/12/13  17:20 
# @file   ：test.py.PY
# @Tool   ：PyCharm
import heapq

t = [1,0,1,2,4,7]

def find_sum(stones):
      # dp[i] 容量为i的背包能装入的最大石头重量
    total, l = sum(stones), len(stones)
    dp = [0] * (total // 2 + 1)
    state = [ [0 for _ in range(total // 2 + 1)] for _ in range(total // 2 + 1)]
      # 对于每块石头，依次判断他们是否被放入J个背包所产生的变化
    for i in range(0, l):
        # 使用倒序，在计算当前容量为j的背包时，需要用到比其容量小的背包的信息。
        # 如果使用正序，需要用二维数组 第一维来记录上一轮状态、
        for j in range(total // 2, stones[i] - 1, -1):
            # 两种选择 1 不装此石头，则当前容量为j的背包装入的最大石头重量不变
            # 2 装入此石头，这当前容量为j的背包的最大石头重量应当为 容量为j-stones[i]的背包装入的最大石头重量+当前石头重量
           # dp[j] = max(dp[j], dp[j - stones[i]] + stones[i])
            if dp[j] < dp[j - stones[i]] + stones[i]:
                dp[j] = dp[j - stones[i]] + stones[i]
                # 第i块石头在容量为j的背包中被选择了
                state[i][j] = 1
    print(state)
    return total - 2 * dp[-1]

def scheduleCourse(courses):
    courses.sort(key=lambda c: c[1])

    q = list()
        # 优先队列中所有课程的总时间
    total = 0
    print(courses)
    for ti, di in courses:
        print(q)
        if total + ti <= di:
            total += ti
                # Python 默认是小根堆
            heapq.heappush(q, -ti)
        elif q and -q[0] > ti:
            total -= -q[0] - ti
            heapq.heappop(q)
            heapq.heappush(q, -ti)

    return len(q)




if __name__ == '__main__':
    print(find_sum(t))
    courses = [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
    scheduleCourse(courses)