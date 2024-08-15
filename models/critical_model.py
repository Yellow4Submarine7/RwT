# models/critical_model.py

from sentence_transformers import SentenceTransformer, util

class CriticalModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def compute_similarity(self, question, entity, possible_answers):
        """
        Compute semantic similarity between the entity and possible answers,
        considering the context of the question.
        """
        question_entity = f"Question: {question} Entity: {entity}"
        embeddings = self.model.encode([question_entity] + possible_answers)
        
        similarities = util.cosine_similarity(embeddings[0], embeddings[1:])
        return similarities.mean().item()
    
    def evaluate(self, question, entity, possible_answers):
        """
        Evaluate if the entity contextually fits the question by assigning a score
        based on semantic suitability.
        """
        similarity_score = self.compute_similarity(question, entity, possible_answers)
        return similarity_score