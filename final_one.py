# 假设20个供应点（产地m个地方）、10个需求点（销售n个地方），
    # 首先需要   数组   建立个供需平衡表矩阵（20+1+1）*（10+1+1）
    # 第1到n列代表需求点（销售处），第n+1列代表供应点的产量
    # 第1到m行代表供应点（产地处），第m+1行代表需求点的需求量（销量）
    # 供应点预设为20  m=20
import numpy as np
import math
import sys
import random
from tkinter import *
from scipy import linalg

class IOT_factory:
    def __init__(self):
        self.x = 0
        self.y = 0
# 列表最小、次小的差额计算
def first_seconday_sort(array):
    # 最小值是min_value；次小值是min_value2，下面的语句是对于min_value两个进行初始化
    if array[0]>array[1]:
        min_value2 = array[0]
        min_value = array[1]
    else:
        min_value = array[0]
        min_value2 = array[1]
    # 对于整个给定的一列或者一行数组进行遍历，
    for i in range(2, len(array)-3):
        if array[i]<=min_value:
            min_value2 = min_value
            min_value = array[i]
        elif array[i]<=min_value2:
            min_value2 = array[i]
    return min_value2-min_value

# 计算罚数，将计算值填入bal_s矩阵
    # 给一个矩阵m+2*n+2、填入最边的一行和一列罚数
def fill_penalty(bal_s):
    row = bal_s.shape[0]
    column = bal_s.shape[1]
    if row<=3 or column <=3:
        return bal_s
    else:
        for i in range(1,row-2):
            temp = first_seconday_sort(bal_s[i-1])
            bal_s[i-1,column-2] = temp
        for i in range(1,column-2):
            temp = first_seconday_sort(bal_s[:,i-1])
            bal_s[row-2,i-1] = temp
        return bal_s

# 返回最大罚数类型、坐标
    # （行列两个里面的最大值，并返回相应的数组下标）    返回值为tag = [1/0, np.argmax(bal_s[？？？？])]
        # 返回tag ，标记1则是最后一行的列罚数；标记0则是最后列的行罚数；
        # 如果最后一行的列罚数大于等于最后一列的行罚数，那么就返回列罚数的下标（row-1，第几列）
def find_penalty(bal_s):
    row = bal_s.shape[0]
    column = bal_s.shape[1]
    if np.max(bal_s[row-2])>=np.max(bal_s[:,column-2]):
        tag = [1, np.argmax(bal_s[row-2])]
        return tag
    else:
        tag = [0, np.argmax(bal_s[:,column-2])]
        return tag

# 根据罚数进行运输调整
    # 确定最大罚数所在位置后，根据其所在位置先寻找对应的行或者列的最小运价、然后进行抵消操作，消除一行或者一列
def delete_line(bal_s, tag, bal_s_traffic):
    row = bal_s.shape[0]
    column = bal_s.shape[1]
    # 根据最大罚数（可能是行的或者列的）来寻找对应的最小运价
    if tag[0]==1:
        # tag[1]就是最大的列罚数在第row-1行中的数组下标(即第几列确定了)，现在对于该列进行最小运价寻找，寻找到的坐标
        price_min_column = tag[1]
        target = bal_s[:,tag[1]]
        price_min_row = np.argmin(target[0:row-3])
    else:
        # tag[1]就是最大的行罚数在第column-1行中的数组下标(即第几行确定了)，现在对于该行进行最小运价寻找，寻找到的坐标
        price_min_row = tag[1]
        target = bal_s[tag[1]]
        price_min_column = np.argmin(target[0:column-3])

    # 根据找到的最小运价来确定是“”供>需“”还是”“供<需“”
    # 填入供量平衡表（带运量，无罚数）
    r_row = int(bal_s[price_min_row, column-1]) - 1
    r_column = int(bal_s[row-1, price_min_column]) - 1

    if bal_s[price_min_row, column-3] > bal_s[row-3, price_min_column]:
        # 填入供量平衡表（带运量，无罚数）
        temp = bal_s[row-3, price_min_column]
        bal_s_traffic[r_row, r_column] = temp
        # 供大于需 , 供=供-需
        bal_s[price_min_row, column-3]-=bal_s[row-3, price_min_column]
        # 删除需求列
        bal_s = np.delete(bal_s, price_min_column, axis=1)
    elif bal_s[price_min_row, column-3] < bal_s[row-3, price_min_column]:
        # 填入供量平衡表（带运量，无罚数）
        temp = bal_s[price_min_row, column-3]
        bal_s_traffic[r_row, r_column] = temp
        # 供小于需 , 需=需-供
        bal_s[row-3, price_min_column]-=bal_s[price_min_row, column-3]
        # 删除供应行
        bal_s = np.delete(bal_s, price_min_row, axis=0)
    else:
        # 填入供量平衡表（带运量，无罚数）
        temp = bal_s[row-3, price_min_column]
        bal_s_traffic[r_row, r_column] = temp
        # 供等于需
        bal_s[price_min_row, column-3]-=bal_s[row-3, price_min_column]
        # 删除需求列并且补0进行防止退化处理
        bal_s = np.delete(bal_s, price_min_column, axis=1)
        bal_s[price_min_row, 0] = 0

    return bal_s

# 供需平衡调整函数（只调用一次！！）
    # (开始循环前、保证供需平衡、对于不平衡的供需进行填充,首先得到产量和&销量和，进行比较)
def adjust_sup_dem(bal_s):
    # print("倒数第二列代表产量，产量和：",end="")
    sum_supply = sum(bal_s[:,bal_s.shape[1]-3])
    # print("倒数第二行代表销量，销量和：",end="")
    sum_demand = sum(bal_s[bal_s.shape[0]-3])
    # 情况1:产量等于销量
    if sum_supply == sum_demand:
    # 情况2:产量大于销量
        pass
    elif sum_supply > sum_demand:
        # 要在本来是m=10行，n=10列的运价部分中，添加一列假设需求列表（全为0）
        # 插入 的链表共有 m+3 个元素
        insert_column = np.zeros(bal_s.shape[0])
        # 插入位置是原来的倒数第3列前
        bal_s = np.insert(bal_s, bal_s.shape[1]-3, values=insert_column, axis=1)
        # 产大于销量，所以补上假设的销量   sum_supply - sum_demand，补上的位置那一列的倒数第4个
        bal_s[bal_s.shape[0]-3,bal_s.shape[1]-4] = int(sum_supply - sum_demand)
    # 情况3:产量小于销量
    else :
        # 要在本来是m=10行，n=10列的运价部分中，添加一行假设供应点的列表（全为0）
        # 插入 的链表共有 n+3 个元素
        insert_column = np.zeros(bal_s.shape[1])
        # 插入位置是原来的倒数第3行前
        bal_s = np.insert(bal_s, bal_s.shape[0]-3, values=insert_column, axis=0)
        # 产大于销量，所以补上假设的销量   sum_supply - sum_demand，补上的位置那一列的倒数第2个
        bal_s[bal_s.shape[0]-4,bal_s.shape[1]-3] =  int(sum_demand - sum_supply)
    return bal_s

# 返回检验数矩阵
def check_number(bal_s_backup, bal_s_traffic):
    backup_m = int(bal_s_backup.shape[0])
    backup_n = int(bal_s_backup.shape[1])
    price_tag = 1
    while price_tag!=0:
        price_tag =0
        # 求括号里面行最大
        for i in range(1, backup_m-2):
            bal_s_backup[i-1,backup_n-1] = -10000
            for j in range(1, backup_n-2):
                if bal_s_traffic[i-1,j-1]!=0:
                    if bal_s_backup[i-1,j-1]>bal_s_backup[i-1,backup_n-1]:
                        bal_s_backup[i-1,backup_n-1] = int(bal_s_backup[i-1,j-1])
        # 求括号里面列最小
        for j in range(1, backup_n-2):
            bal_s_backup[backup_m-1,j-1] = 10000
            for i in range(1, backup_m-2):
                if bal_s_traffic[i-1,j-1]!=0:
                    if bal_s_backup[i-1,j-1]<bal_s_backup[backup_m-1,j-1]:
                        bal_s_backup[backup_m-1,j-1] = int(bal_s_backup[i-1,j-1])
        # kongjio 矩阵计计算 先判断一下 是否全为0
        for i in range(1, backup_m-2):
            backup_flag = 0
            for k in range(1, backup_n-2):
                if bal_s_traffic[i-1,k-1]!=0:
                    if bal_s_backup[i-1,k-1]!=0:
                        backup_flag = 1
            if backup_flag!=0:
                for j in range(1, backup_n-2):
                    bal_s_backup[i-1,j-1] = bal_s_backup[i-1,j-1] - bal_s_backup[i-1,backup_n-1]

        for i in range(1, backup_m-2):
            backup_flag = 0
            for k in range(1, backup_n-2):
                if bal_s_traffic[i-1,k-1]!=0:
                    if bal_s_backup[i-1,k-1]!=0:
                        backup_flag = 1
            if backup_flag!=0:
                for j in range(1, backup_n-2):
                    bal_s_backup[i-1,j-1] = bal_s_backup[i-1,j-1] - bal_s_backup[backup_m-1,j-1]
        # 判断是否计算完成
        for i in range(1, backup_m-2):
            for j in range(1, backup_n-2):
                if bal_s_traffic[i-1,j-1]>0:
                    if bal_s_backup[i-1,j-1]!=0:
                        price_tag = 1
                        break
    return bal_s_backup

# 闭回路算法 B[I,J]
def close_loop(bal_s_traffic, L, K):
    M = bal_s_traffic.shape[0] - 3
    N = bal_s_traffic.shape[1] - 3
    close_loop_B = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            # 筛选矩阵填入1
            if bal_s_traffic[i, j]!=0:
                close_loop_B[i, j]=1
    Z1 = np.zeros(M*N*1000).astype(np.intc)
    Z2 = np.zeros(M*N*1000).astype(np.intc)
    KK = np.zeros(M*N*1000).astype(np.intc)
    W = 1
    Z1[W] = L
    Z2[W] = K
    j = 1
    while j!=N+1:
        if close_loop_B[L, j-1]==1:
            if j-1!=K :
                W = W + 1
                Z1[W] = L
                Z2[W] = j-1
                KK[W] = 1
        j = j + 1
    # print("W:{}".format(W))
    I = 2
    YYKK_flag = 0
    while YYKK_flag != 1:
        I1 = Z1[I]
        J1 = Z2[I]
        # 如果Z1[K[I]]==Z1[I]，进行列搜索，插入树
        if Z1[KK[I]]==Z1[I]:
            x = 1
            while x-1 != M:
                if close_loop_B[x-1, J1]==1:
                    if x != I1:
                        W = W + 1
                        Z1[W] = x-1
                        Z2[W] = J1
                        KK[W] = I
                        x = x + 1
                    else:
                        x = x + 1
                else:
                    x = x + 1
        else :
            y = 1
            while y-1 != N:
                if close_loop_B[I1, y-1]==1:
                    if y-1 == J1:
                        y = y + 1
                    else:
                        W = W + 1
                        Z1[W] = I1
                        Z2[W] = y-1
                        KK[W] = I
                        if y-1==K:
                            YYKK_flag = 1
                            break
                        else:
                            y = y + 1
                else:
                    y = y + 1
        I = I + 1
    # print(W)
    answer = []
    while W!=1:
        # print("Z1[W]:",Z1[W],"Z2[W]:",Z2[W])
        answer.append(Z1[W])
        answer.append(Z2[W])
        answer.append(bal_s_traffic[Z1[W], Z2[W]])
        W = KK[W]
    return answer

# 返回指定矩阵内最小的元素和下标
def find_check(bal_s_check):
    tag = [0,0,0]
    m = bal_s_check.shape[0]
    n = bal_s_check.shape[1]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if tag[0] > bal_s_check[i-1, j-1]:
                tag[0] = bal_s_check[i-1, j-1]
                tag[1] = i-1
                tag[2] = j-1
    return tag

print("-----需求-供应数据生成ING-----")
# 序列i由1->10来命名工厂—————初始化，全部归零————bal_s----balanced_schedule平衡表
names = locals()
bal_s = np.zeros([13,13])
# 平衡表基本数据生成


for i in range(1,11):
    names['supply'+str(i)] = IOT_factory()
    names['supply'+str(i)].x = random.randint(1, 100)
    names['supply'+str(i)].y = random.randint(1, 100)

for i in range(1,11):
    names['demand'+str(i)] = IOT_factory()
    names['demand'+str(i)].x = random.randint(1, 100)
    names['demand'+str(i)].y = random.randint(1, 100)

for i in range(1,11):
    # 工厂&需求点
    bal_s[i-1, 10] = random.randint(1, 1000)
    bal_s[10, i-1] = random.randint(1, 1000)
    # bal_s[i-1, 12] = int(i)
    # bal_s[12, i-1] = int(i)
    # 工厂与需求点的距离&运价
    for j in range(1,11):
        # bal_s[i-1, j-1] = random.randint(1, 100)
        X = names['supply'+str(i)].x - names['demand'+str(j)].x
        Y = names['supply'+str(i)].y - names['demand'+str(j)].y
        bal_s[i-1, j-1] = math.floor(math.sqrt(X**2 + Y**2))+1

print("-----需求-供应数据生成成功-----")
# 存储bal_s，为bal_s_backup备用
bal_s_backup = bal_s
# 供需平衡调整
bal_s = adjust_sup_dem(bal_s)
# 静态行列数标注
for i in range(1,bal_s.shape[0]-2):
    bal_s[i-1, bal_s.shape[1]-1] = i
for i in range(1,bal_s.shape[1]-2):
    bal_s[bal_s.shape[0]-1, i-1] = i
print("-----调整供需平衡后的数据-----")
print(bal_s)

# 备份一个供需平衡表（带运价，无罚数）
bal_s_backup = adjust_sup_dem(bal_s_backup)

# 备份一个供量平衡表（带运量，无罚数）
bal_s_traffic = np.zeros(bal_s.shape)
bal_s_traffic[bal_s.shape[0]-3] = bal_s[bal_s.shape[0]-3]
bal_s_traffic[:,bal_s.shape[1]-3] = bal_s[:,bal_s.shape[1]-3]
# 存储动态的运量的坐标和数值（需要处理）

print("-----生成初始解-----")
# 生成初始解
# —bal_s_traffic（运量矩阵） bal_s_backup是备份矩阵
while bal_s.shape[0]!=4 or bal_s.shape[1]!=4:
    bal_s = fill_penalty(bal_s)
    tag = find_penalty(bal_s)
    bal_s = delete_line(bal_s, tag, bal_s_traffic)
if bal_s[0, 1]==bal_s[1, 0]:
    bal_s_traffic[int(bal_s[0, 3]-1), int(bal_s[3, 0]-1)] = bal_s[0, 1]
print(bal_s_traffic)
print("-----生成检验数的矩阵-----")
# 将bal_s_check 处理成只有检验数的矩阵
bal_s_check = check_number(bal_s_backup, bal_s_traffic)
for i in range(1,4):
    bal_s_check = np.delete(bal_s_check, bal_s_check.shape[0]-1, axis=0)
    bal_s_check = np.delete(bal_s_check, bal_s_check.shape[1]-1, axis=1)
# print(bal_s_check)
print("-----最小检验数数值&下标-----")
check_check = find_check(bal_s_check)
# print(check_check)

while check_check[0]<0:
    print("-----闭回路求出 x y value-----")
    ANS = close_loop(bal_s_traffic, check_check[1], check_check[2])
    # print(ANS)
    ANS_min = ANS[2]
    print("-----运量修正量-----")
    for i in range(0,len(ANS),6):
        if ANS[i+2]<ANS_min:
            ANS_min = ANS[i+2]
    # print(ANS_min)
    print("-----traffic修正-----")
    # 空格坐标补入
    ANS.append(check_check[1])
    ANS.append(check_check[2])
    ANS.append(0)
    # bal_s_traffic
    ANS_flag = -1
    for i in range(0,len(ANS),3):
        bal_s_traffic[ANS[i],ANS[i+1]] = ANS[i+2] + ANS_flag*ANS_min
        ANS_flag = ANS_flag*(-1)
    # print(bal_s_traffic)
    # print(bal_s_backup)
    print("-----生成检验数的矩阵-----")
    # 将bal_s_check 处理成只有检验数的矩阵
    bal_s_check = check_number(bal_s_backup, bal_s_traffic)
    for i in range(1,4):
        bal_s_check = np.delete(bal_s_check, bal_s_check.shape[0]-1, axis=0)
        bal_s_check = np.delete(bal_s_check, bal_s_check.shape[1]-1, axis=1)
    # print(bal_s_check)
    print("-----最小检验数数值&下标-----")
    check_check = find_check(bal_s_check)
    # print(check_check)

print("-----生成最优解-----")
print(bal_s_traffic)
print("-----最小检验数数值&下标-----")
print(check_check)
print("-----待处理矩阵-----")
print(bal_s_backup)
print("Completed!")
for i in range(1,11):
    print("names['supply'+str(i)]:", names['supply'+str(i)].x,"&", names['supply'+str(i)].y)

for i in range(1,11):
    print("names['demand'+str(i)]:", names['demand'+str(i)].x,"&", names['demand'+str(i)].y)
# mainloop()
