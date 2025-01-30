from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile
from agents.Group80.ResistanceEvaluation import evaluate_board_resistance
import math
import numpy as np
from collections import deque
from queue import PriorityQueue

class AlphaBetaAgent(AgentBase):
    def __init__(self, colour):
        super().__init__(colour)
        self.colour = colour
        self.depth = 3
        self.swap_check = False

    def make_move(self, turn: int, board: Board, opponent_move: Move | None):
        """
        Determines the next move for the agent, including swap logic if the opponent's move
        matches predefined coordinates.
        """
        # Red coordinates for swap
        red_marked_positions = {
            # Red edges formed with an obtuse angle
            (0, 10), (1, 9), (1, 10), (9, 0), (10, 0), (9, 1),

            # Central red positions
            (5, 5), (5, 4), (5, 6), (4, 5), (4, 6), (6, 4), (6, 5)
        }

        # Check for swap move
        if (not self.swap_check and 
            self.colour == Colour.BLUE and 
            opponent_move is not None and 
            (opponent_move.x, opponent_move.y) in red_marked_positions):
            
            self.swap_check = True  # Mark swap check as complete
            return Move(-1, -1)    # Perform swap move
        
        self.swap_check = True

        # Red minimizing player
        if self.colour == Colour.RED:
            return self.minimax(board, self.depth, False, -math.inf, math.inf)[1]

        # Blue maximizing player
        return self.minimax(board, self.depth, True, -math.inf, math.inf)[1]
        
    def minimax(self, board: Board, depth, is_maximizing, alpha, beta) -> tuple[int, Move]:
        
        # Terminal condition: Depth is 0 or the game has ended
        if depth == 0 or board.has_ended(Colour.BLUE) or board.has_ended(Colour.RED):
            eval_score = self.evaluate_board(board)  # Heuristic evaluation
            return eval_score, None

        best_move = None

        # Generate moves using two-distance heuristic
        moves = self.twoDistanceRankAndPrune(board, Colour.BLUE if is_maximizing else Colour.RED)
        
        if is_maximizing:
            max_eval = -math.inf

            for move in moves:
                # Play the move
                board.set_tile_colour(move.x, move.y, Colour.BLUE)

                # Recursively call alpha-beta search
                eval_score, _ = self.minimax(board, depth - 1, False, alpha, beta)

                # undo move
                board.set_tile_colour(move.x, move.y, None)

                # Update evaluation
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                # Alpha-Beta pruning
                alpha = max(alpha, max_eval)

                if alpha >= beta:
                    break

            return max_eval, best_move

        # Minimizing player
        else:
            min_eval = math.inf

            for move in moves:
                # Play the move
                board.set_tile_colour(move.x, move.y, Colour.RED)

                # Recursively call alpha-beta search
                eval_score, _ = self.minimax(board, depth - 1, True, alpha, beta)

                # undo move
                board.set_tile_colour(move.x, move.y, None)

                # Update evalution
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                # Alpha-Beta pruning
                beta = min(beta, min_eval)

                if alpha >= beta:
                    break

            return min_eval, best_move

    def evaluate_board(self, board: Board) -> float:
        ''' Returns a score for the board state. '''
        return evaluate_board_resistance(board)
    
    def get_neighbors(self, x:int, y:int, board: Board) -> list[Tile]:
        """
        Calculate the neighbors of a tile at (x, y) on the board.
        
        Args:
            x (int): x coord of the tile
            y (int): y coord of the tile
            board (list[list[Tile]]): A 2D list representing the Hex board.
            
        Returns:
            list[Tile]: A list of neighboring Tile objects.
        """
        neighbors = []
        
        for k in range(Tile.NEIGHBOUR_COUNT):  # Iterate through all 6 potential neighbors
            nx = x + Tile.I_DISPLACEMENTS[k]
            ny = y + Tile.J_DISPLACEMENTS[k]
            
            # Check if the neighbor is within the board boundaries
            if 0 <= nx < board.size and 0 <= ny < board.size:
                neighbors.append(board.tiles[nx][ny])
        
        return neighbors

    def twoDistanceForBoardPositions(self, board: Board, player_colour: Colour, rotateView: bool):
        # Initialize data structures using numpy
        isTwoDistance = np.zeros((board.size, board.size), dtype=bool)
        twoDistance = np.full((board.size, board.size), math.inf)

        # Initialize BFS queue
        queue = deque()

        if player_colour == Colour.RED:  # Red connects top to bottom
            if rotateView:
                # Reverse the view: bottom-to-top
                for x in range(board.size):
                    for y in range(board.size):
                        if x == board.size - 1:  # Bottom row
                            queue.append((x, y))
                            twoDistance[x, y] = 0
                            isTwoDistance[x, y] = True
            else:
                # Normal view: top-to-bottom
                for x in range(board.size):
                    for y in range(board.size):
                        if x == 0:  # Top row
                            queue.append((x, y))
                            twoDistance[x, y] = 0
                            isTwoDistance[x, y] = True

        else:  # Blue connects left to right
            if rotateView:
                # Reverse the view: right-to-left
                for y in range(board.size):
                    for x in range(board.size):
                        if y == board.size - 1:  # Right column
                            queue.append((x, y))
                            twoDistance[x, y] = 0
                            isTwoDistance[x, y] = True
            else:
                # Normal view: left-to-right
                for y in range(board.size):
                    for x in range(board.size):
                        if y == 0:  # Left column
                            queue.append((x, y))
                            twoDistance[x, y] = 0
                            isTwoDistance[x, y] = True

        # BFS to calculate two-distance
        while queue:
            x, y = queue.popleft()
            current_distance = twoDistance[x, y]

            # Get neighbors of the current cell
            for neighbor in self.get_neighbors(x, y, board):
                nx, ny = neighbor.x, neighbor.y

                if not isTwoDistance[nx, ny]:  # If the neighbor not visited
                    isTwoDistance[nx, ny] = True

                    # Update distance based on cell type
                    if neighbor.colour == player_colour:
                        twoDistance[nx, ny] = current_distance
                    elif neighbor.colour is None:
                        twoDistance[nx, ny] = current_distance + 1

                    # Add neighbor to the queue
                    queue.append((nx, ny))

        return twoDistance
    
    def twoDistanceRankAndPrune(self, board: Board, modeColour: Colour):
        """
        Implements the TwoDistanceRankAndPrune algorithm.
        
        Args:
            board: Current state of the Hex board.
            modeColour: The colour (player) for which the evaluation is performed.
            depth: Depth of the search in the game tree.
        
        Returns:
            List of sorted moves.
        """

        # Step 1: Compute distances for both players
        redWinDistance1 = self.twoDistanceForBoardPositions(board, Colour.RED, rotateView=True)
        redWinDistance2 = self.twoDistanceForBoardPositions(board, Colour.RED, rotateView=False)
        blueWinDistance1 = self.twoDistanceForBoardPositions(board, Colour.BLUE, rotateView=True)
        blueWinDistance2 = self.twoDistanceForBoardPositions(board, Colour.BLUE, rotateView=False)

        # Step 2: Initialize priority queue
        queue = PriorityQueue()
        
        # Step 3: Calculate scores and enqueue nodes
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour != None: # Skip occupied cells
                    continue

                # Calculate scores
                redScore = redWinDistance1[i][j] + redWinDistance2[i][j]
                blueScore = blueWinDistance1[i][j] + blueWinDistance2[i][j]
                finalScore = min(redScore, blueScore)
                
                queue.put((finalScore, (i, j)))

        # Step 4: Prune and sort moves
        sortedList = []
        fanOut = 20  # Limit number of moves considered

        while queue.queue and len(sortedList) < fanOut:
            _, (x, y) = queue.get()  # Dequeue the lowest-score move
            sortedList.append(Move(x, y))
        
        return sortedList
