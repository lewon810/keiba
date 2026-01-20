import random

def predict(race_data):
    """
    Takes race data (list of dicts) and returns a formatted string of predictions.
    """
    if not race_data:
        return "No data to predict."
        
    # Simple random prediction logic (shuffling)
    # in a real app, this would use machine learning or heuristics
    
    # Make it deterministic based on the first horse's name
    if race_data:
        seed_val = sum(ord(c) for c in race_data[0].get('name', ''))
        random.seed(seed_val)
    
    predictions = list(race_data)
    random.shuffle(predictions)
    
    result_lines = ["Prediction Ranking:"]
    for i, horse in enumerate(predictions):
        name = horse.get("name", "Unknown")
        jockey = horse.get("jockey", "")
        # symbol 
        symbol = "  "
        if i == 0: symbol = "◎ "
        elif i == 1: symbol = "○ "
        elif i == 2: symbol = "▲ "
        elif i == 3: symbol = "△ "
        elif i == 4: symbol = "☆ "
        
        line = f"{symbol} {i+1}. {name} ({jockey})"
        result_lines.append(line)
        
    return "\n".join(result_lines)
