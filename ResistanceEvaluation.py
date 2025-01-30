from src.Board import Board
from src.Colour import Colour
import numpy as np
import heapq

def create_resistance_matrix(board: Board, colour: Colour):
    """
    Create a resistance matrix for the board based on the player's perspective.
    """
    resistance = np.zeros((board.size, board.size))

    for i in range(board.size):
        for j in range(board.size):

            if board.tiles[i][j].colour == None:  # Empty cell
                resistance[i][j] = 1

            elif board.tiles[i][j].colour == colour:  # Player's piece
                resistance[i][j] = 0

            else:  # Opponent's piece
                resistance[i][j] = float('inf')

    return resistance

def neighbors(x, y, size):
    """
    Get the neighboring cells of a given cell on the board.
    """
    I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
    J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]

    for k in range(6):
        nx = x + I_DISPLACEMENTS[k]
        ny = y + J_DISPLACEMENTS[k]
        
        if 0 <= nx < size and 0 <= ny < size:
            yield nx, ny

def calculate_resistance(resistance_matrix, start_boundary, end_boundary):
    """
    Use Dijkstra's algorithm to calculate total resistance between boundaries.
    """
    size = len(resistance_matrix)
    dist = np.full((size, size), float('inf'))
    pq = []  # Priority queue for Dijkstra's

    # Initialize distances for the start boundary
    for x, y in start_boundary:
        dist[x][y] = resistance_matrix[x][y]
        heapq.heappush(pq, (dist[x][y], x, y))

    # Dijkstra's algorithm
    while pq:
        curr_dist, x, y = heapq.heappop(pq)
        if curr_dist > dist[x][y]:
            continue

        for nx, ny in neighbors(x, y, size):
            new_distance = curr_dist + resistance_matrix[nx][ny]
            if new_distance < dist[nx][ny]:  # Relaxation
                dist[nx][ny] = new_distance
                heapq.heappush(pq, (new_distance, nx, ny))
    
    # Find minimum distance to the end boundary
    return min([dist[x][y] for x, y in end_boundary])

def evaluate_board_resistance(board: Board) -> float:
    """
    Evaluate the current board state.
    """
    # Create resistance matrices
    resistance_matrix_red = create_resistance_matrix(board, Colour.RED)
    resistance_matrix_blue = create_resistance_matrix(board, Colour.BLUE)

    # Define boundaries (fixed board of size 11x11)
    start_boundary_red = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10)]
    end_boundary_red = [(10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10)]

    start_boundary_blue = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0)]
    end_boundary_blue = [(0, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10)]

    # Calculate resistances
    resistance_red = calculate_resistance(resistance_matrix_red, start_boundary_red, end_boundary_red)
    resistance_blue = calculate_resistance(resistance_matrix_blue, start_boundary_blue, end_boundary_blue)

    # Compute evaluation value
    if resistance_red == 0:
        return 0  # Red wins
    if resistance_blue == 0:
        return float('inf')  # Blue wins
    
    # Red would minimize this; Blue would maximize
    return resistance_red / resistance_blue