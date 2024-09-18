import math
import random

from tqdm import tqdm

random.seed(42)

class Action:
    def __init__(self, from_state, to_state):
        self.from_state = from_state
        self.to_state = to_state
        self.q_num = 0
        self.q_den = 0
    
    def get_q(self):
        return 0 if self.q_den == 0 else (self.q_num / self.q_den)
    
    def update_q(self, q):
        self.q_num += q
        self.q_den += 1

class State:
    def __init__(self, id, reward):
        self.id = id
        self.reward = reward
        self.actions = []
    
    def add_action(self, action):
        self.actions.append(action)

    def choose_action(self, epsilon, is_train, is_on_policy):
        selected_action = self.actions[int(random.random() * len(self.actions))]
        if (not is_train) or (is_on_policy and (random.random() >= epsilon)):
            for action in self.actions:
                if action.get_q() > selected_action.get_q():
                    selected_action = action
        return selected_action

class Environment:
    def cell_id(self, x, y):
        return (x * self.m) + y

    def state_id(self, x, y, cycle_pos, config):
        return (((self.cell_id(x, y) * (self.cycle_length)) + cycle_pos) * self.num_configs) + config
    
    def is_goal(self, x, y, config):
        return (x, y, config) == (self.init_cell[0], self.init_cell[1], 0)
    
    def is_ghost(self, x, y, cycle_pos):
        for ghost in self.ghosts:
            cell = ghost[cycle_pos % self.cycle_length]
            if (x, y) == cell:
                return True
        return False

    def is_dot(self, x, y, config):
        p = config
        for dot in self.dots:
            if (x, y) == dot and (p % 2):
                return True
            p //= 2
        return False
    
    def next_config(self, x, y, config):
        p = config
        delta = 1
        for dot in self.dots:
            if (x, y) == dot and (p % 2):
                return config - delta
            p //= 2
            delta *= 2
        return config

    def define_states(self):
        for x in range(self.n):
            for y in range(self.m):
                for cycle_pos in range(self.cycle_length):
                    ghost_cell = self.is_ghost(x, y, cycle_pos)

                    for config in range(self.num_configs):
                        curr_state_id = self.state_id(x, y, cycle_pos, config)
                        reward = -1

                        if self.is_goal(x, y, config):
                            self.terminal_states.add(curr_state_id)
                            reward = 200
                        elif ghost_cell:
                            self.terminal_states.add(curr_state_id)
                            reward = -100
                        elif self.is_dot(x, y, config):
                            reward = 100

                        self.states.append(State(curr_state_id, reward))

    def define_actions(self):
        for x in range(self.n):
            for y in range(self.m):
                if not self.grid[x][y]:
                    continue

                left_empty = bool(y and self.grid[x][y - 1])
                up_empty = bool(x and self.grid[x - 1][y])

                for cycle_pos in range(self.cycle_length):
                    for config in range(self.num_configs):
                        curr_state_id = self.state_id(x, y, cycle_pos, config)
                        
                        if left_empty:
                            state_pair = (self.states[curr_state_id], self.states[self.state_id(x, y - 1, (cycle_pos + 1) % self.cycle_length, self.next_config(x, y, config))])
                            state_pair[0].add_action(Action(state_pair[0], state_pair[1]))

                            state_pair = (self.states[self.state_id(x, y - 1, cycle_pos, config)], self.states[self.state_id(x, y, (cycle_pos + 1) % self.cycle_length, self.next_config(x, y - 1, config))])
                            state_pair[0].add_action(Action(state_pair[0], state_pair[1]))
                        
                        if up_empty:
                            state_pair = (self.states[curr_state_id], self.states[self.state_id(x - 1, y, (cycle_pos + 1) % self.cycle_length, self.next_config(x, y, config))])
                            state_pair[0].add_action(Action(state_pair[0], state_pair[1]))
                            
                            state_pair = (self.states[self.state_id(x - 1, y, cycle_pos, config)], self.states[self.state_id(x, y, (cycle_pos + 1) % self.cycle_length, self.next_config(x - 1, y, config))])
                            state_pair[0].add_action(Action(state_pair[0], state_pair[1]))

    def __init__(self, alpha, epsilon, is_on_policy, grid, ghosts, dots, init_cell):
        self.alpha = alpha
        self.epsilon = epsilon
        self.is_on_policy = True

        self.grid = grid
        self.ghosts = ghosts
        self.dots = dots

        self.n = len(grid)
        self.m = len(grid[0])

        self.cycle_length = 0
        for ghost in ghosts:
            self.cycle_length = math.gcd(self.cycle_length, len(ghost))
        self.num_configs = 2 ** len(dots)

        self.states = []
        self.terminal_states = set()
        self.init_cell = init_cell

        self.define_states()
        self.define_actions()

        init_state_id = self.state_id(init_cell[0], init_cell[1], 0, self.num_configs - 1)
        self.init_state = self.states[init_state_id]

    def break_state(self, state_id):
        cell_id, cycle_config = state_id // (self.cycle_length * self.num_configs), state_id % (self.cycle_length * self.num_configs)
        return (cell_id // self.m, cell_id % self.m, cycle_config // self.num_configs, cycle_config % self.num_configs)
    
    def one_episode(self, max_length, is_train):
        state = self.init_state

        state_path = [state]
        action_path = []
        for _ in range(max_length):
            action = state.choose_action(self.epsilon, is_train, self.is_on_policy)
            action_path.append(action)

            state = action.to_state
            state_path.append(state)

            if state.id in self.terminal_states:
                break

            broken_state = self.break_state(state.id)
            if (broken_state[0], broken_state[1], broken_state[3]) == (init_cell[0], init_cell[1], 0):
                break
        
        if not is_train:
            display = []
            for state in state_path:
                y = self.break_state(state.id)
                display.append([y[0], y[1]])
            print(display)
        
        reward = 0
        for action in action_path[::-1]:
            reward = (reward * self.alpha) + action.to_state.reward
            action.update_q(reward)
    
    def train(self, episodes, max_length):
        for _ in tqdm(range(episodes)):
            self.one_episode(max_length, True)

grid = [
    [True,  True,   True,   True,   True,   True],
    [False,  True,   False,   True,   False,   True],
    [True,  True,   False,   True,   True,   True],
    [True,  False,   False,   True,   False,   True],
    [True,  True,   True,   True,   True,   True],
    [False,  False,   True,   False,   False,   True]
]

ghosts = [
    [(4, 0), (4, 1), (4, 0), (3, 0)],
    [(2, 4), (2, 5), (2, 4), (2, 3)]
]
dots = [(1, 3), (2, 1), (5, 2), (3, 0)]
init_cell = (5, 5)

env = Environment(0.9, 0.1, True, grid, ghosts, dots, init_cell)
env.train(10000, 10000)
env.one_episode(100, False)