import time
import random
import numpy as np
import torch as th


# 모든 요소의 값이 0으로 채워진 임의의 MxN 더미행렬 생성(결과 저장용)
def matrix_null(rows, columns):
    matrix = [ [0 for i in range(columns)] for j in range(rows)]
    return matrix

# 각 요소의 값이 100 이상 1000 미만의 랜덤값으로 채워진 임의의 MxN 행렬 생성
def matrix_init(rows, columns):
    matrix = [ [random.randrange(100, 1000) for i in range(columns)] for j in range(rows)]
    return matrix


# 방법1: 중첩 반복문을 이용한 연산
def matrix_mul_loop(X, Y):
    result = matrix_null(len(X), len(Y[0]))

    start_time = time.time()
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]
    finish_time = time.time()

    cpu_time = finish_time - start_time
    print("{:.5f}초 on CPU / 중첩 반복문".format(cpu_time))

# 방법2: 리스트 컴프리헨션을 이용한 연산
def matrix_mul_list(X, Y):
    result = matrix_null(len(X), len(Y[0]))

    start_time = time.time()
    result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
    finish_time = time.time()

    cpu_time = finish_time - start_time
    print("{:.5f}초 on CPU / 리스트 컴프리헨션".format(cpu_time))

# 방법3: Numpy를 이용한 연산
def matrix_mul_numpy(X, Y):
    result = matrix_null(len(X), len(Y[0]))
    X_numpy = np.array(X)
    Y_numpy = np.array(Y)

    start_time = time.time()
    result = np.dot(X_numpy, Y_numpy)
    finish_time = time.time()

    cpu_time = finish_time - start_time
    print("{:.5f}초 on CPU / Numpy".format(cpu_time))

# 방법4: Pytorch를 이용한 연산
def matrix_mul_pytorch(X, Y):
    result = matrix_null(len(X), len(Y[0]))
    X_tensor = th.FloatTensor(X)
    Y_tensor = th.FloatTensor(Y)

    if th.cuda.is_available():
        # CPU에서 연산
        start_time = time.time()
        result = th.mm(X_tensor, Y_tensor)
        finish_time = time.time()

        cpu_time = finish_time - start_time
        print("{:.5f}초 on CPU / Pytorch".format(cpu_time))

        # GPU에서 연산
        x = X_tensor.to("cuda")
        y = Y_tensor.to("cuda")

        start_time = time.time()
        results = th.mm(x, y)
        finish_time = time.time()

        gpu_time = finish_time - start_time
        print("{:.5f}초 on GPU / Pytorch".format(gpu_time))

    else:
        print("액세스 할 수 있는 GPU 모듈이 존재하지 않습니다.")


if __name__ == '__main__':
    X = matrix_init(500, 500)
    Y = matrix_init(500, 500)
    # X와 Y의 사이즈 조절하며 연산시간 비교

    matrix_mul_loop(X, Y)
    matrix_mul_list(X, Y)
    matrix_mul_numpy(X, Y)
    matrix_mul_pytorch(X, Y)