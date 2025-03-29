#pylint: disable=all
import json
print("Loading multi modal embedder") # GOnna take a some time...
from multi_modal_embedder import MultimodalEmbedder

def zero_shot(query, **kwargs):
    print("Initializing multimodel embedder")
    embedder = MultimodalEmbedder()
    embedder.load_indices()

    print("Making a search")
    result = embedder.search(query=query, **kwargs)

    # Pretty print
    json_formatted_str = json.dumps(result, indent=2)
    print(json_formatted_str)
    print("You should get json with lots of ")


if __name__=="__main__":
    zero_shot(query={"text": "Quantum Mechanics"})
