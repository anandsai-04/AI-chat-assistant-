import json
import os
from typing import List, Dict

# We use the standard ChatML / OpenAI format which is what most SLMs expect.
def create_conversation(user_text: str, assistant_text: str) -> Dict:
    return {
        "messages": [
            {
                "role": "system", 
                "content": "You are an elite expert specializing in Coding, Fitness, Diet, and Spiritual Guidance."
            },
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
    }

def generate_synthetic_data() -> List[Dict]:
    """Generates synthetic training data for our specific domains."""
    data = []
    
    # Domain: Coding
    data.append(create_conversation(
        "How do I structure a FastAPI project?",
        "For a production FastAPI project, you should use an Application Factory pattern, separate your routes into an `app/api/routes` directory, and strictly isolate business logic into an `app/services` layer. Use Dependency Injection for passing database clients or settings."
    ))
    data.append(create_conversation(
        "Why use Pydantic v2?",
        "Pydantic v2 is written in Rust, making it up to 50x faster than v1. It provides strict type validation, excellent integration with FastAPI, and clear JSON Schema generation."
    ))

    # Domain: Diet & Disease
    data.append(create_conversation(
        "I have a mild fever and throat pain. What diet helps?",
        "Based on nutritional guidelines, focus on hydration and easily digestible foods. Drink warm broths, herbal teas with honey (to soothe the throat), and eat plain foods like BRAT (Bananas, Rice, Applesauce, Toast). Avoid dairy and heavy fats, as they can irritate the throat or be difficult to digest."
    ))

    # Domain: Fitness
    data.append(create_conversation(
        "What is a good workout for building back muscle safely?",
        "To build back muscle safely, focus on compound pulling movements with strict form. Pull-ups or Lat Pulldowns target the latissimus dorsi. Barbell or Dumbbell Rows target the rhomboids and mid-back. Always maintain a neutral spine, brace your core, and do not jerk the weight."
    ))

    # Domain: Spiritual Guidance
    data.append(create_conversation(
        "How can I find peace when I feel overwhelmed?",
        "Many spiritual traditions teach that peace is found by returning to the present moment. Through mindfulness or meditation, observe your breath without trying to change it. Acknowledge your overwhelming feelings without judgment, letting them pass like clouds in the sky. As Thich Nhat Hanh says, 'Smile, breathe, and go slowly.'"
    ))
    data.append(create_conversation(
        "What is the meaning of letting go?",
        "Letting go does not mean not caring. In spiritual literature, letting go means detaching from the outcome and accepting reality as it is, rather than how you wish it to be. It is the surrender of the ego's need to control the universe."
    ))

    return data

def save_jsonl(data: List[Dict], filename: str):
    """Saves a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
if __name__ == "__main__":
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_data()
    
    # For a real project, we would split into train and valid (e.g., 80/20)
    train_data = dataset[:-1] # All but the last
    valid_data = dataset[-1:] # Just the last one for validation
    
    save_jsonl(train_data, "data/train.jsonl")
    save_jsonl(valid_data, "data/valid.jsonl")
    
    print(f"Generated {len(train_data)} training examples and {len(valid_data)} validation examples.")
    print("Data saved to data/train.jsonl and data/valid.jsonl")
