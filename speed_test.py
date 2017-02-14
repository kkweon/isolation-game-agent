import random
import timeit
import argparse

from collections import namedtuple
from isolation import Board
from sample_players import improved_score
from game_agent import custom_score, CustomPlayer

def test_depth(agent1, agent2, NUM=10):
    result = [[], []]
    for _ in range(NUM):
        result = compare_round(agent1, agent2, result)
    return result

def compare_round(agent1, agent2, result):
    result_1 = compare_depth(agent1, agent2)
    result_2 = compare_depth(agent2, agent1)

    return [result[0] + result_1[0] + result_2[1], result[1] + result_1[1] + result_2[0]]

def compare_depth(agent1, agent2):
    game = Board(agent1, agent2)
    result = [[], []]

    for _ in range(2):
        move = random.choice(game.get_legal_moves())
        game.apply_move(move)

    curr_time_millis = lambda: 1000 * timeit.default_timer()
    time_limit = 150
    for _ in range(10): 
        move_start = curr_time_millis()
        time_left = lambda : time_limit - (curr_time_millis() - move_start)

        legal_moves = game.get_legal_moves(agent1)
        loc, i = agent1.get_move(game, legal_moves, time_left, test=True)
        if i==0:
            break 

        if i > 1000:
            print(i)
        game.apply_move(loc)
        result[0].append(i)

        move_start = curr_time_millis()
        time_left = lambda : time_limit - (curr_time_millis() - move_start)

        legal_moves = game.get_legal_moves(agent2)
        loc, i = agent2.get_move(game, legal_moves, time_left, test=True)
        if i == 0:
            break
        game.apply_move(loc)
        result[1].append(i)

    return result

def print_result(list_):
    assert len(list_) == 2

    msg = """Agent 1: Avg Depth:{}, Max Depth: {}, Min Depth: {}, Len: {}
Agent 2: Avg Depth:{}, Max Depth: {}, Min Depth: {}, Len: {}
""".format(sum(list_[0])/len(list_[0]), max(list_[0]), min(list_[0]), len(list_[0]), \
           sum(list_[1])/len(list_[1]), max(list_[1]), min(list_[1]), len(list_[1]))
    print("")
    print(msg)
    print("")
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of round matches", default=10, type=int)
    args = parser.parse_args()

    Agent = namedtuple('Agent', ['player', 'name'])
    CUSTOM_ARGS = {"method": 'alphabeta', 'iterative': True}
    test_agents = [CustomPlayer(score_fn=improved_score, **CUSTOM_ARGS), CustomPlayer(score_fn=custom_score, **CUSTOM_ARGS)]
    result = test_depth(*test_agents, args.n)
    msg = print_result(result)
    with open('speed_test_result.txt', 'w') as f:
        f.write(msg)


if __name__ == '__main__':
    main()

