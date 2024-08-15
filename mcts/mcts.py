# mcts/mcts.py

import random
from typing import Callable
from mcts.node import MCTSNode

class MCTS:
    def __init__(self, policy_fn: Callable, value_fn: Callable, num_simulations: int = 100):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.num_simulations = num_simulations

    def search(self, root_state):
        root = MCTSNode(state=root_state)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.is_fully_expanded():
                node = node.select_child()
                search_path.append(node)

            # Expansion
            parent = search_path[-1]
            actions, priors = self.policy_fn(parent.state)
            parent.expand(actions, priors)
            
            if parent.children:
                node = random.choice(parent.children)
                search_path.append(node)

            # Evaluation
            value = self.value_fn(node.state)

            # Backpropagation
            for node in reversed(search_path):
                node.update(value)

        return root

    def get_action_probabilities(self, state, temperature=1):
        root = self.search(state)
        visits = [child.visits for child in root.children]
        actions = [child.action for child in root.children]
        
        if temperature == 0:
            best_action = actions[visits.index(max(visits))]
            probs = [0] * len(actions)
            probs[actions.index(best_action)] = 1
            return actions, probs
        
        visits = [v ** (1 / temperature) for v in visits]
        total = sum(visits)
        probs = [v / total for v in visits]
        
        return actions, probs

    def select_action(self, state, temperature=1):
        actions, probs = self.get_action_probabilities(state, temperature)
        return random.choices(actions, weights=probs)[0]