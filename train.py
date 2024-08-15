# train.py

import torch
from models.rollout_model import RolloutModel, train_rollout_model
from models.critical_model import CriticalModel
from models.policy_model import PolicyModel
from mcts.mcts import MCTS
from kg.knowledge_graph import KnowledgeGraph
from kg.query import KGQuery
from utils.data_loader import load_dataset
from tqdm import tqdm

def generate_training_data(mcts, kg_query, questions, num_simulations=100):
    training_data = []
    for question in tqdm(questions, desc="Generating training data"):
        root_state = {"current_entity": question["seed_entity"], "question": question["question"]}
        mcts.search(root_state)  # This will populate the search tree
        
        # Extract training examples from the search tree
        def extract_examples(node):
            if node.visits > 0:
                state_value = node.value / node.visits
                training_data.append((node.state, state_value))
                for child in node.children:
                    extract_examples(child)
        
        extract_examples(mcts.root)
    
    return training_data

def train(config):
    # Initialize models
    policy_model = PolicyModel(config.policy_model_name)
    rollout_model = RolloutModel(config.rollout_model_name)
    critical_model = CriticalModel(config.critical_model_name)
    
    # Load knowledge graph and create query interface
    kg = KnowledgeGraph()
    kg.load_from_file(config.kg_file)
    kg_query = KGQuery(kg)
    
    # Load dataset
    train_questions = load_dataset(config.train_file)
    
    # Initialize MCTS
    def policy_fn(state):
        actions = kg_query.get_possible_actions(state["current_entity"])
        probs = policy_model.get_action_probabilities(state, actions)
        return actions, list(probs.values())
    
    def value_fn(state):
        rollout_value = rollout_model(state).item()
        critical_value = critical_model.evaluate(state["question"], state["current_entity"], 
                                                 kg_query.get_context(state["current_entity"])["entities"])
        return (1 - config.lambda_param - config.mu_param) * rollout_value + \
               config.lambda_param * critical_value
    
    mcts = MCTS(policy_fn, value_fn, num_simulations=config.num_simulations)
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # Generate training data using MCTS
        training_data = generate_training_data(mcts, kg_query, train_questions)
        
        # Train rollout model
        train_rollout_model(rollout_model, training_data, epochs=config.rollout_epochs, lr=config.rollout_lr)
        
        # Update MCTS value function
        def updated_value_fn(state):
            rollout_value = rollout_model(state).item()
            critical_value = critical_model.evaluate(state["question"], state["current_entity"], 
                                                     kg_query.get_context(state["current_entity"])["entities"])
            return (1 - config.lambda_param - config.mu_param) * rollout_value + \
                   config.lambda_param * critical_value
        
        mcts.value_fn = updated_value_fn
    
    # Save trained models
    torch.save(rollout_model.state_dict(), config.rollout_model_save_path)
    torch.save(critical_model.state_dict(), config.critical_model_save_path)

if __name__ == "__main__":
    from config import TrainingConfig
    train(TrainingConfig())