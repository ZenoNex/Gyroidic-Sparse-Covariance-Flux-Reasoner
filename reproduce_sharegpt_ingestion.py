
import json

def test_preprocess_sharegpt():
    print("Testing Preprocessing Logic on ShareGPT format...")
    
    # Mock ShareGPT sample
    sample = {
        "id": "12345",
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there."}
        ]
    }
    
    # Logic from dataset_ingestion_system.py _preprocess_sample ('text' mode)
    text_fields = ['text', 'content', 'body', 'description', 'title']
    text_content = ""
    
    print(f"Sample keys: {sample.keys()}")
    
    # Standard fields
    for field in text_fields:
        if field in sample and sample[field]:
            text_content += str(sample[field]) + "\n"
    
    # ShareGPT format (conversations)
    if 'conversations' in sample and isinstance(sample['conversations'], list):
        for turn in sample['conversations']:
            if isinstance(turn, dict):
                role = turn.get('from', 'unknown')
                value = turn.get('value', turn.get('text', ''))
                if value:
                    text_content += f"{role}: {value}\n"
    
    # Alpaca format (instruction/input/output)
    if 'instruction' in sample:
        text_content += f"Instruction: {sample['instruction']}\n"
        if sample.get('input'):
            text_content += f"Input: {sample['input']}\n"
        if sample.get('output'):
            text_content += f"Output: {sample['output']}\n"
    
    if not text_content.strip():
        print("❌ Result: None (Dropped)")
    else:
        print(f"✅ Result: {text_content.strip()}")

if __name__ == "__main__":
    test_preprocess_sharegpt()
