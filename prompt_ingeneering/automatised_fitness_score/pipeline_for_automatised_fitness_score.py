import spacy

nlp = spacy.load("en_core_web_sm")

def extract_names(text):
    # Process the text with SpaCy
    doc = nlp(text)
    
    # Collect all entities labeled as "PERSON"
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Return the unique names (optional, to avoid duplicates)
    return list(set(names))

# Example usage
story = """
Once upon a time, in a small cottage nestled in the heart of an enchanted forest, there lived two siblings: Emily, the tender-hearted girl, and Tommy, the energetic boy. Emily was known for her calm nature and unwavering compassion towards all living creatures.
"""


names_in_story = extract_names(story)
print(names_in_story)
