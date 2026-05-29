import math
import random
import time
from collections import defaultdict


class GameState:
    def __init__(self, board=None, current_player=1):
        self.board = board if board else [0] * 9
        self.current_player = current_player

    def get_legal_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def make_move(self, move):
        new_board = self.board[:]
        new_board[move] = self.current_player
        return GameState(new_board, -self.current_player)

    def is_terminal(self):
        return self.get_winner() is not None or not self.get_legal_moves()

    def get_winner(self):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        return None

    def get_utility(self, player):
        w = self.get_winner()
        if w == player:
            return 1
        elif w == -player:
            return -1
        return 0

    def display(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for i in range(0, 9, 3):
            print(' '.join(symbols[self.board[j]] for j in range(i, i+3)))


def minimax(state, player):
    if state.is_terminal():
        return state.get_utility(player), None

    if state.current_player == player:
        best_val = -math.inf
        best_move = None
        for move in state.get_legal_moves():
            val, _ = minimax(state.make_move(move), player)
            if val > best_val:
                best_val, best_move = val, move
        return best_val, best_move
    else:
        best_val = math.inf
        best_move = None
        for move in state.get_legal_moves():
            val, _ = minimax(state.make_move(move), player)
            if val < best_val:
                best_val, best_move = val, move
        return best_val, best_move


def alpha_beta(state, player, alpha=-math.inf, beta=math.inf):
    if state.is_terminal():
        return state.get_utility(player), None

    if state.current_player == player:
        best_val = -math.inf
        best_move = None
        for move in state.get_legal_moves():
            val, _ = alpha_beta(state.make_move(move), player, alpha, beta)
            if val > best_val:
                best_val, best_move = val, move
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = math.inf
        best_move = None
        for move in state.get_legal_moves():
            val, _ = alpha_beta(state.make_move(move), player, alpha, beta)
            if val < best_val:
                best_val, best_move = val, move
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move


def heuristic_eval(state, player):
    w = state.get_winner()
    if w == player:
        return 100
    elif w == -player:
        return -100

    score = 0
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for line in lines:
        vals = [state.board[i] for i in line]
        p_count = vals.count(player)
        o_count = vals.count(-player)
        if o_count == 0:
            score += 10 ** p_count
        if p_count == 0:
            score -= 10 ** o_count
    return score


def heuristic_alpha_beta(state, player, depth, alpha=-math.inf, beta=math.inf):
    if state.is_terminal() or depth == 0:
        return heuristic_eval(state, player), None

    if state.current_player == player:
        best_val = -math.inf
        best_move = None
        for move in state.get_legal_moves():
            val, _ = heuristic_alpha_beta(state.make_move(move), player, depth - 1, alpha, beta)
            if val > best_val:
                best_val, best_move = val, move
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = math.inf
        best_move = None
        for move in state.get_legal_moves():
            val, _ = heuristic_alpha_beta(state.make_move(move), player, depth - 1, alpha, beta)
            if val < best_val:
                best_val, best_move = val, move
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = state.get_legal_moves()

    def ucb1(self, c=1.414):
        if self.visits == 0:
            return math.inf
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        return max(self.children, key=lambda n: n.ucb1())

    def expand(self):
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        child = MCTSNode(self.state.make_move(move), parent=self, move=move)
        self.children.append(child)
        return child

    def rollout(self, root_player):
        state = self.state
        while not state.is_terminal():
            moves = state.get_legal_moves()
            state = state.make_move(random.choice(moves))
        return state.get_utility(root_player)

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


def mcts(state, player, iterations=1000):
    root = MCTSNode(state)
    for _ in range(iterations):
        node = root
        while not node.state.is_terminal() and not node.untried_moves:
            node = node.select_child()
        if not node.state.is_terminal() and node.untried_moves:
            node = node.expand()
        result = node.rollout(player)
        node.backpropagate(result)

    if not root.children:
        return None, None
    best = max(root.children, key=lambda n: n.visits)
    return best.wins / best.visits if best.visits > 0 else 0, best.move


def run_tests():
    print("=" * 50)
    print("ASSIGNMENT 1 — SEARCH ALGORITHMS")
    print("=" * 50)

    print("\n--- Test 1: Minimax on empty board ---")
    s = GameState()
    val, move = minimax(s, 1)
    print(f"Minimax recommends move: {move}, value: {val}")

    print("\n--- Test 2: Alpha-Beta on same state ---")
    t0 = time.time()
    val_ab, move_ab = alpha_beta(GameState(), 1)
    t1 = time.time()
    val_mm, move_mm = minimax(GameState(), 1)
    t2 = time.time()
    print(f"Alpha-Beta: move={move_ab}, val={val_ab}, time={t1-t0:.4f}s")
    print(f"Minimax:    move={move_mm}, val={val_mm}, time={t2-t1:.4f}s")
    assert val_ab == val_mm, "Alpha-Beta and Minimax should agree on value"
    print("Values match — Alpha-Beta correct.")

    print("\n--- Test 3: Alpha-Beta on near-terminal state (X about to win) ---")
    board = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    state = GameState(board, 1)
    val, move = alpha_beta(state, 1)
    print(f"X should block/win by playing move: {move}, value: {val}")
    assert move == 2, f"Expected move 2 (winning), got {move}"
    print("Correct — plays winning move.")

    print("\n--- Test 4: Heuristic Alpha-Beta (depth=4) ---")
    val_h, move_h = heuristic_alpha_beta(GameState(), 1, depth=4)
    print(f"Heuristic AB: move={move_h}, heuristic value={val_h}")

    print("\n--- Test 5: MCTS (1000 iterations) ---")
    val_m, move_m = mcts(GameState(), 1, iterations=1000)
    print(f"MCTS recommends move: {move_m}, win rate: {val_m:.3f}")

    print("\n--- Test 6: Game simulation — Alpha-Beta (X) vs MCTS (O) ---")
    state = GameState()
    move_count = 0
    while not state.is_terminal():
        if state.current_player == 1:
            _, move = alpha_beta(state, 1)
            label = "X (AB)"
        else:
            _, move = mcts(state, -1, iterations=500)
            label = "O (MCTS)"
        if move is None:
            break
        print(f"{label} plays: {move}")
        state = state.make_move(move)
        move_count += 1

    state.display()
    winner = state.get_winner()
    print(f"Winner: {'X' if winner == 1 else 'O' if winner == -1 else 'Draw'}")

    print("\n--- Test 7: Terminal state utility check ---")
    win_board = [1, 1, 1, -1, -1, 0, 0, 0, 0]
    s = GameState(win_board, -1)
    assert s.is_terminal()
    assert s.get_utility(1) == 1
    assert s.get_utility(-1) == -1
    print("Terminal state utility — correct.")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()