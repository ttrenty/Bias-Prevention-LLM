import json
import requests

def update_json_file(filename, new_data, nested_list =False):
    ## Modifier car quand génère des histoires les mets dans une liste
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("File not found. Creating a new one.")
        data = []
    except json.JSONDecodeError:
        print("File is empty or invalid. Initializing with an empty list.")
        data = []
    if nested_list:
        data[len(data)-1].append(new_data)
    else :
        data.append(new_data)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)



def LLM_request(prompt) : 

    url = 'http://localhost:1234/v1/chat/completions'
    # url = "https://api.hyperbolic.xyz/v1/chat/completions"

    input_text = prompt
    payload = {
        "messages": [{"role": "user", "content": input_text}],
        # "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # "model": "gpt-3.5-turbo"  # Specify the model (if needed)
    }
    headers = {
        'Content-Type': 'application/json',  
        'Accept': 'application/json',        
    }
    """
    headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzeWxhZm9udEBlbnNjLmZyIiwiaWF0IjoxNzM2NDM5NjA0fQ.bulegzRLNecWwQyNS9Tdtjf89ftPrgy7KXAA7og3arA"
}
    """
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json() 
        model_response = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        return model_response
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)



def retrieve_story(input_string): ### PROVISOIRE ICI
    for key in input_string:
        last_key = key

    return input_string[key]
    
    """
    keyword = "STORY:"
    index_last = input_string.rfind(keyword)
    if index_last == -1:
        index_last=input_string.rfind("STORY*")
    index_last_comment = input_string.rfind("COMMENT")
    
    if index_last_comment == -1 or index_last_comment<index_last :
        index_last_comment = input_string.rfind("Comment:")
        if index_last_comment == -1 or index_last_comment<index_last :
            index_last_comment == len(input_string)
    
    if index_last != -1:
        result = input_string[index_last + len(keyword):index_last_comment].strip()
        return result
    else :
        print("Non splitable history.")
        return -1
    """
