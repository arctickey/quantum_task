from src.task_1.solution import count_islands

def test_case_1():
    matrix = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 1]
    ]
    assert count_islands(matrix) == 2

def test_case_2():
    matrix = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ]
    assert count_islands(matrix) == 3

def test_case_3():
    matrix = [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]
    assert count_islands(matrix) == 2
