# models/policy_model.py

import openai
from typing import List, Dict

class PolicyModel:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def get_action_probabilities(self, state: Dict, possible_actions: List[str]) -> Dict[str, float]:
        """
        Given the current state and possible actions, return the probability distribution over actions.
        
        :param state: A dictionary representing the current state
        :param possible_actions: A list of possible actions (relations) to choose from
        :return: A dictionary mapping actions to their probabilities
        """
        prompt = self._construct_prompt(state, possible_actions)
        response = self._query_llm(prompt)
        return self._parse_response(response, possible_actions)

    def _construct_prompt(self, state: Dict, possible_actions: List[str]) -> str:
        """Construct a prompt for the LLM based on the current state and possible actions."""
        prompt = f"Current entity: {state['current_entity']}\n"
        prompt += f"Question: {state['question']}\n"
        prompt += "Possible relations:\n"
        for action in possible_actions:
            prompt += f"- {action}\n"
        prompt += "\nBased on the current entity and the question, assign probabilities to each relation for the next step. Respond with a JSON object mapping relations to probabilities."
        return prompt

    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt."""
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an AI assistant helping with knowledge graph reasoning."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str, possible_actions: List[str]) -> Dict[str, float]:
        """Parse the LLM's response into a probability distribution over actions."""
        try:
            probabilities = eval(response)  # Assuming the LLM responds with a valid Python dict
            # Normalize probabilities
            total = sum(probabilities.values())
            return {action: prob / total for action, prob in probabilities.items() if action in possible_actions}
        except:
            # If parsing fails, return uniform distribution
            uniform_prob = 1.0 / len(possible_actions)
            return {action: uniform_prob for action in possible_actions}