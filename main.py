# main.py

from models.rollout_model import RolloutModel
from models.critical_model import CriticalModel
from models.policy_model import PolicyModel
from mcts.mcts import MCTS
from kg.knowledge_graph import KnowledgeGraph
from kg.query import KGQuery
from utils.data_loader import load_dataset
from utils.evaluation import evaluate_performance
from config import InferenceConfig

def main(config):
    # Load models
    policy_model = PolicyModel(config.policy_model_name)
    rollout_model = RolloutModel(config.rollout_model_name)
    rollout_model.load_state_dict(torch.load(config.rollout_model_path))
    critical_model = CriticalModel(config.critical_model_name)
    
    # Load knowledge graph and create query interface
    kg = KnowledgeGraph()
    kg.load_from_file(config.kg_file)
    kg_query = KGQuery(kg)
    
    # Load test dataset
    test_questions = load_dataset(config.test_file)
    
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
    
    # Inference
    results = []
    for question in test_questions:
        root_state = {"current_entity": question["seed_entity"], "question": question["question"]}
        mcts.search(root_state)
        
        # Select the best action based on visit counts
        best_action = max(mcts.root.children, key=lambda c: c.visits).action
        final_entity = kg_query.execute_action(root_state["current_entity"], best_action)[0]
        
        results.append({
            "question": question["question"],
            "predicted_answer": final_entity,
            "true_answer": question["answer_entities"]
        })
    
    # Evaluate performance
    performance = evaluate_performance(results)
    print("Performance:", performance)

if __name__ == "__main__":
    main(InferenceConfig())