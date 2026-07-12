import os
import json
import asyncio
import re
from pathlib import Path
from app.llm.clients.ollama import OllamaClient

async def convert_markdown_to_qa():
    """Reads Markdown files and uses a local LLM to generate synthetic Q&A pairs for fine-tuning."""
    
    input_dir = Path("data/raw/train")
    output_file = Path("data/train.jsonl")
    
    if not input_dir.exists():
        print(f"Directory {input_dir} not found.")
        return
        
    md_files = list(input_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {input_dir}. Please add some small documentation files to train on.")
        return
        
    print(f"Found {len(md_files)} markdown files. Initializing LLM for synthetic generation...")
    # We use our local Ollama client. (If this is too slow on your machine, you can swap this to use Groq/Gemini).
    llm = OllamaClient()
    
    generated_data = []
    
    for file_path in md_files:
        print(f"Processing {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # We prompt the LLM to act as a data generation pipeline
        prompt = [
            {
                "role": "system",
                "content": "You are a synthetic data generation pipeline. Read the text and extract 3 highly technical Question and Answer pairs based on the text. Output strictly as a JSON array of objects with 'question' and 'answer' keys. No other text."
            },
            {
                "role": "user",
                "content": f"Text:\n{content[:2500]}" # Limiting length to ensure it fits in a local SLM context window
            }
        ]
        
        try:
            response = await llm.chat(messages=prompt)
            # Clean up potential markdown formatting around the JSON
            cleaned_response = re.sub(r'```json\n|\n```|```', '', response).strip()
            
            qa_pairs = json.loads(cleaned_response)
            
            for pair in qa_pairs:
                if "question" in pair and "answer" in pair:
                    generated_data.append({
                        "messages": [
                            {"role": "system", "content": "You are an elite Domain Expert."},
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]}
                        ]
                    })
                    
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            print(f"Raw response was: {response}")

    if generated_data:
        # Append to the jsonl file
        with open(output_file, "a", encoding="utf-8") as f:
            for item in generated_data:
                f.write(json.dumps(item) + "\n")
        print(f"✅ Generated {len(generated_data)} synthetic Q&A pairs and appended to {output_file}")
    else:
        print("No valid Q&A pairs were generated.")

if __name__ == "__main__":
    asyncio.run(convert_markdown_to_qa())
