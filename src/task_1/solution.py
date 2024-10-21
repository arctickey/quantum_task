def count_islands(matrix: list[list[int]]) -> int:
    """
    Counts the number of islands in a given matrix, where 1 represents land (island) 
    and 0 represents water (ocean).

    Args:
    matrix (List[List[int]]): A 2D list representing the map with islands (1) and ocean (0).

    Returns:
    int: The total number of islands.
    """
    def dfs(x: int, y: int) -> None:
        if x < 0 or y < 0 or x >= len(matrix) or y >= len(matrix[0]) or matrix[x][y] == 0:
            return
        matrix[x][y] = 0  # Sink the island
        
        dfs(x + 1, y)
        dfs(x - 1, y)
        dfs(x, y + 1)
        dfs(x, y - 1)

    if not matrix:
        return 0

    island_count = 0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1: 
                island_count += 1
                dfs(i, j)

    return island_count