
import json
from utils import update_json_file
from openai import OpenAI

def new_request_llm(dic, prompt):
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="hermes-3-llama-3.1-8b"
    )

    # Define the conversation with the AI
    messages = [
        {"role": "system", "content": "You are a creative storyteller that aims at creating stories without stereotypes. Do not sacrifice story quality or length for the sake of brevity in the JSON structure. Your stories should be at least 350 words long."},
        {'role': 'user', 'content': prompt}
    ]

    character_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "story",
            "schema": {
                "type": "object",
                "properties": {
                    "story": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": 
                            dic,
                            "required": [key for key in dic]  
                        },
                        "minItems": 1,
                        "maxItems": 1
                    }
                },
                "required": ["story"]
            },
        }
    }
    response = client.chat.completions.create(
        model="hermes-3-llama-3.1-8b",
        messages=messages,
        response_format=character_schema,
    #  max_tokens=2048
    )
    for key in dic:
        last_index_dic = key

    raw_answer = json.loads(response.choices[0].message.content)
    story = raw_answer["story"][0][last_index_dic]

    return story, raw_answer["story"][0]
    # results = json.loads(response.choices[0].message.content)
    # print(json.dumps(results, indent=2))

def generate_story_from_prompt(task_prompts, story_prompts, nb_generation, config_prompts, pass_prompt=[]):

    ### VERIFIER QUE L'HISTOIRE FINALE NE CONTIENT PAS QUE DES PERSONNAGES NON BINAIRES
    ### SI C'EST LE CAS REGENERER L'HISTOIRE
    for z, (task_prompt, dic) in enumerate(zip(task_prompts, config_prompts)) :
        if z in pass_prompt :
            print("Stories for prompt ", z, "already generated so we go directly to next prompt")
            continue
        list_story_for_one_prompt = []
        for story_prompt in story_prompts:
            list_stories_for_one_story_prompt =[]
            for i in range(10):
                intput_prompt = story_prompt+"\n"+ task_prompt+ "\n"
                # print(intput_prompt)
                story_returned=""
                while " he " not in story_returned.lower() and " she " not in story_returned.lower() and " his " not in story_returned.lower() and " her " not in story_returned.lower()  : # Vérifier que l'histoire ne contient pas que des persos non binaires
                    story_returned, full_answer = new_request_llm(dic, intput_prompt)
                    #print(story_returned.split())
                list_stories_for_one_story_prompt.append(full_answer)
    
                # print("successfully added story")
            print("Go to next story")
            list_story_for_one_prompt.append(list_stories_for_one_story_prompt)
        update_json_file("new_stories_"+str(nb_generation)+".json", list_story_for_one_prompt)
        print("Passing to next prompts :", z)


def Generate_story(nb_generation):
    
    with open("new_stories_"+ str(nb_generation) + ".json", "r") as file:
        stories = json.load(file)

    with open("prompt_generation_"+ str(nb_generation) + "_copy.json", "r") as file:
        task_prompts = json.load(file)

    with open("dic_initial_prompt.json", "r") as f :   
        config_prompts = json.load(f)
    
    with open("prompt_for_biaised_story.json", "r") as file:
        biased_story_prompt = json.load(file)
    #generate_story_from_prompt(initial_prompt, biased_story_prompt, nb_generation, config_prompts, pass_prompt=[0])
    rewrite_specific_story(stories, task_prompts, biased_story_prompt, config_prompts)

# Generate_story(1)

def rewrite_specific_story(stories, task_prompts, story_prompts, config_prompts):
    for z, (task_prompt, dic, list_list_story) in enumerate(zip(task_prompts, config_prompts, stories)) :
        for j,(story_prompt, list_story) in enumerate(zip(story_prompts ,list_list_story)):
            for i, dic_story in enumerate(list_story):
                for key in dic_story:
                    last_index =key
                old_dic =dic_story
                story_returned = dic_story[last_index]
                # print(len(story_returned.split()))
                while len(story_returned.split()) < 200 or (" he " not in story_returned.lower() and " she " not in story_returned.lower() and " his " not in story_returned.lower() and " her " not in story_returned.lower())  : # Vérifier que l'histoire ne contient pas que des persos non binaires 
                    intput_prompt = story_prompt+"\n"+ task_prompt+ "\n"
                    story_returned, dic_story = new_request_llm(dic, intput_prompt)
                stories[z][j][i] = dic_story
                if old_dic != dic_story :
                    print("modif apportée")
                else :
                    print("Tout était ok de base")
            print("Go to next story")
        print("Passing to next prompts :", z)
    
    #with open("corrected_story.json", 'w') as file:
        # json.dump(stories, file, indent=4)


Generate_story(1)



