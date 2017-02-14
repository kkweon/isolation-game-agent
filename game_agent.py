"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import numpy as np

#######################################
# HEURISTIC FUNCTION HELPER FUNCTIONS #
#######################################


def is_valid_move(move, MAX_ROW, MAX_COL):
    """ Check if move is out of the board size

    Args:
        move (tuple): (row, col)
        MAX_ROW (int): The length of vertical axis of the board size
        MAX_COL (int): The length of horizontal axis of the board size

    Returns:
        True or False
    """
    r, c = move
    if r < 0 or r >= MAX_ROW:
        return False
    if c < 0 or c >= MAX_COL:
        return False
    return True


def get_hueristic(row, col, MAX_VAL=10, discount=0.9, directions=[(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]):
    """Returns custom value matrix, value

    value[row][col] implies how good move (row, col) is

    Args:
        row (int): Row size of the board
        col (int): Col size of the board
        MAX_VAL (int): The highest value (usually center of the board)
        discount (float): Starting from the best value, nearest cells will receive discounted value
        directions (list): Possible legal moves

    Returns:
        value (2 by 2 matrix): value[row][col] = some value 

    """
    center_point = (row // 2, col // 2)
    value = [[0.0 for c in range(col)] for r in range(row)]
    change = True
    while change:
        change = False
        for r in range(len(value)):
            for c in range(len(value[0])):
                if (r, c) == center_point:
                    if value[r][c] != MAX_VAL:
                        value[r][c] = MAX_VAL
                        change = True
                else:
                    near_points = [(r + delta_r, c + delta_c) for delta_r,
                                   delta_c in directions if is_valid_move((r + delta_r, c + delta_c), row, col)]
                    max_near = max(value[x][y] for x, y in near_points)
                    if max_near * discount != value[r][c]:
                        value[r][c] = max_near * discount
                        change = True

    return value

# Define the VALUE here such that it doesn't have to compute multiple times
VALUE = get_hueristic(row=7, col=7, MAX_VAL=10, discount=0.9)


def get_value(value, move):
    """Returns value[x][y]
    where x,y = move

    Args:
        value (2 by 2 matrix): Value matrix
        move (tuple): (x, y)

    Returns:
        float: value[x][y]
    """
    x, y = move
    return value[x][y]


def custom_score_v1(game, player):
    """ Heuristic Function Version 1

    (1): len(my moves) - len(opponent moves)
    (2): average of my future moves heuristics

    Returns (1) + (2)
    """
    global VALUE

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    all_blank_spaces = game.get_blank_spaces()
    row, col = game.height, game.width

    opponent = game.get_opponent(player)
    my_legal_moves = game.get_legal_moves(player=player)
    opponent_legal_moves = game.get_legal_moves(player=opponent)

    my_score = len(my_legal_moves)
    opp_score = len(opponent_legal_moves)
    if len(my_legal_moves) > 0:
        my_score += np.mean([VALUE[r][c] for r, c in my_legal_moves])
    if len(opponent_legal_moves) > 0:
        opp_score += np.mean([VALUE[r][c] for r, c in opponent_legal_moves])

    return my_score - opp_score


def custom_score_v2(game, player):
    """Heuristic Function v2

    (1): len(my moves) - len(opponent moves)
    (2): average value of future moves according to VALUE
    (3): the VALUE of a current position

    Returns (1) + (2) + (3)

    """
    global VALUE

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    all_blank_spaces = game.get_blank_spaces()

    opponent = game.get_opponent(player)
    my_legal_moves = game.get_legal_moves(player=player)
    opponent_legal_moves = game.get_legal_moves(player=opponent)

    my_last_move = game.get_player_location(player=player)
    opponent_last_move = game.get_player_location(player=opponent)

    my_score = len(my_legal_moves) + get_value(VALUE, my_last_move)
    opp_score = len(opponent_legal_moves) + \
        get_value(VALUE, opponent_last_move)
    if len(my_legal_moves) > 0:
        my_score += np.mean([VALUE[r][c] for r, c in my_legal_moves])
    if len(opponent_legal_moves) > 0:
        opp_score += np.mean([VALUE[r][c] for r, c in opponent_legal_moves])

    return my_score - opp_score


def custom_score_v3(a, b, c):

    def custom_score(game, player):

        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("inf")

        blank_spaces = len(game.get_blank_spaces())
        my_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player)))
        return float(my_legal_moves*a / (opponent_legal_moves*b + 1e-5) / (blank_spaces*c + 1e-5))

    return custom_score


def custom_score_KERAS(game, player, model):
    """Use Convolution Neural Network to train the value function.

    V(state) = score

    Notes: 
        - it requires a custom model file which is too big(2gb) to attach
        - it was trained using Keras
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    state = np.array(game.get_board_state()).reshape(-1, 7, 7, 1)
    reward_pred_for_P1 = np.max(model.predict(state))

    if player == game.__player_1__:
        my_reward = reward_pred_for_P1
    else:
        my_reward = -1 * reward_pred_for_P1

    return my_reward


custom_score = custom_score_v3(0.9109905965646489, 0.47001586579228216, 0.16690185540812202)
# custom_score_CNN = custom_score_KERAS


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        NO_MOVE = (-1, -1)
        if len(legal_moves) == 0:
            return NO_MOVE
        loc = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == 'minimax':
                fn = self.minimax
            elif self.method == 'alphabeta':
                fn = self.alphabeta
            else:
                raise Exception("Unknown Method")

            i = 1
            if self.iterative:
                while True:
                    _, loc = fn(game, i)
                    i += 1
            else:
                _, loc = fn(game, self.search_depth)

            return loc

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return loc or NO_MOVE

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            # if depth == 0, no more search down
            # returns current score based on the heuristic function
            return self.score(game, self), []  # game.get_player_location(self)

        # Get all possible move positions of the active player
        # Go deeper by trying out a move and change the player
        # By recursion, it will eventually returns some utility
        # If Maximizing Player: take the maximum value and save whatever the first action brings the max value
        # Else: take the minimum instead and get the action

        legal_moves = game.get_legal_moves(player=game.active_player)

        if len(legal_moves) == 0:
            # If game is done
            # returns utility, and final location (which won't matter)
            return self.score(game, self), []  # game.get_player_location(player=game.inactive_player)
        # Max Function or Min Function
        get_max_or_min = max if maximizing_player else min

        # Initialize best value & action by assigning the worst value
        best_value = float('-inf') if maximizing_player else float('inf')
        best_action = None

        for move in legal_moves:
            # For every possible move, try out each move and get its value
            value, _ = self.minimax(game.forecast_move(move), depth - 1, not maximizing_player)
            # check if the value is "better" than the best_value
            # When maximizing player, a larger value is "better"
            # When minimizing player, a smaller value is "better"
            test_value = get_max_or_min(value, best_value)
            # test_value is the new "better" value
            # if test_value is different from the best value I had before
            # it implies new value is a "better" value
            if test_value != best_value:
                best_value = value
                best_action = move

        return best_value, best_action

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        # "Same as MINIMAX"  BEGINS HERE #
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), []  # game.get_player_location(self)

        legal_moves = game.get_legal_moves(player=game.active_player)

        if len(legal_moves) == 0:
            return self.score(game, self), []  # game.get_player_location(player=game.inactive_player)

        get_max_or_min = max if maximizing_player else min
        best_action = None
        best_value = float('-inf') if maximizing_player else float('inf')

        for move in legal_moves:
            value, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, not maximizing_player)
            test_value = get_max_or_min(best_value, value)
            if test_value != best_value:
                best_value = value
                best_action = move

        # "Same as MINIMAX" ENDS HERE"

            if maximizing_player:
                # Update the lower bound
                # Because I can get at least the current best value
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    # If the current max value is lower than the min value
                    # There is no point in looking further
                    break

            else:
                # Update the upper bound
                # Because my opponent will limit my max value
                beta = min(beta, best_value)
                if beta <= alpha:
                    # If the current max value is lower than the min value
                    # There is no point in looking further
                    break

        return best_value, best_action
