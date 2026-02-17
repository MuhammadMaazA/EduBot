import os 
from openai import OpenAI

OpenAI.api_key ="YOUR_KEY"

os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
MODEL = "gpt-3.5-turbo"

client = OpenAI()

#Generates a response given a prompt using OpenAI's GPT-3 API 
def generate_response(prompt, content_message = "You are a helpful and friendly assistant.Babysitter: easy going, understable language, engaging, filled with curiosity, and with a fantastic plot twist to captivate children", 
                      max_tokens = 1000, n = 1): 
    #MODIFIED : content_message , added the last phrase 
    
    print(prompt)
    print()    
    response = client.chat.completions.create(
        messages =[
            { "role" : "system", "content" : content_message  },
            {"role" : "user", "content" : prompt } ],
        model = MODEL,
        temperature=0.7,
        max_tokens=max_tokens,
        n=n
    )

    return response.choices[0].message.content


def complete_story(story_segment,model=MODEL, temperature=0.7, max_tokens=1000):
    system_prompt = (
        "You are a extremely creative and friendly assistant, responsible with rewriting stories "
        "for a  particular audience: 5-year-old children. Your task is to remodel the story "
        " into an creative new version.This upgraded story should be in the style of  "
        "babysitter: easy going, understable language, engaging, filled with curiosity, and with a fantastic plot twist to "
        "captivate children. Keep the original story, "
        f"and weave the rest of the story from here:\n\n"
        f"'{story_segment}'\n\n"
        f"Continue the story from the end"
        "For example if i provided beginning :"
        "Hello "
        "The output should be in the following way: "
        " Hello there! Let me tell you a fun story about a little girl named Lily who lived in a magical forest." )        

    response = client.chat.completions.create(
        model=model,
        messages= [ {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
  #  return response.choices[0].message.content


#modify generated story 
def mGs(story_segment,indications, model= MODEL, temperature = 0.7, max_tokens = 1000):
    system_prompt = (f"""The goal is to rewrite the story given, with a set of instructions.
    Story: 
    {story_segment}
    Instructions:
    {indications}
    Here is an example of the task:
    
    Story:
    Nur decided to get a dog. She settled on a playful Golden Retriever. She calls her Nurki.
    Indication:
    - Change the ending of the story. 
    - Change the name of her dog. 
    - Change the animal that Nur got to a robot. 
    Generated Story: Nur decided to get a robot. She settled on a helpful RoboAssistant. She calls it  RoboNurki.""")

    response = client.chat.completions.create(
        model=model,
        messages= [
            { "role": "system",
             "content" : system_prompt },
        ],       
        n=3,
        temperature=temperature,
        max_tokens=max_tokens,
    ) 

    return response.choices[0].message.content


#def generate answers based on questions and the story
def gAbQaS(story_segment,question_segment, model= MODEL, temperature = 0.7, max_tokens = 1000):
    system_prompt = (f"""Given the following questions : 
                     ```
                     {question_segment}
                     ```
                     Generate answers that would have answer in the following story {story_segment} 
                     Give explanation to your answers
                     here is an example of what i'm looking for:
                     Story: Today was a sunny day. Nur opened her eyes to her Robot Mini Nur singing a morning song for her.
                     Questions:
                     -What's the name of Nur's Robot ? 
                     -How's the weather today? 
                     -What was Robot doing when Nur woke up
                     -Where did Nur and Tom meet?
                     Generated Answers:
                      - Answer 1: Nur's Robot's name is Mini Nur 
                      - Answer 2: It's sunny day today
                      - Answer 3: When Nur woke up, Robot was singing songs for Nur
                      - Answer 4: Nur didnt meet Tom in the story  """)

    response = client.chat.completions.create(
        model=model,
        messages= [
            { "role": "system",
             "content" : system_prompt },
        ],       
        n=3,
        temperature=temperature,
        max_tokens=max_tokens,
    ) 
    return response.choices[0].message.content

def dSgDaG(topic, age_group, word_count= 200 , model= MODEL, temperature = 0.8, max_tokens = 1000):

    system_prompt = (f""" You are tasked with generating a story based on the following parameters:  
                     1. **Topic** : 
                     {topic}
                     2. **Target Age Group**: 
                     {age_group}
                     3. ** Word Count Limit**:        
                    {word_count} words
                    Make sure the story is age-appropriate, engaging and tailered to the specified word limit. 

                    here is an example of what i'm looking for:
                    {chooseTarget(age_group)}          

                    Ensure the language, tone and content match the developmental level of the age group.
                    Output only the final story
                                    """)

    response = client.chat.completions.create(
        model=model,
        messages= [
            { "role": "system",
             "content" : system_prompt },
        ],       
        n=3,
        temperature=temperature,
        max_tokens=max_tokens,
    ) 
    return response.choices[0].message.content

def complete_story_german(story_segment,model=MODEL, temperature=0.7, max_tokens=1000):
    system_prompt = ("Hier ist der Beginn einer kreativen, freundlichen, fantastischen Geschichte für Kinder. "
            "Behalten Sie den Anfang, fahren Sie bitte auf Französisch fort:"
            f"'{story_segment}'\n\n")

    response = client.chat.completions.create(
        model=model,
        messages= [
            {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return story_segment + response.choices[0].message.content

def chooseTarget(ageGroup):

    if ageGroup=="Toddlers":
        prompt ="""-use of plurals like “dogs”
                -use simple sentences 
                -simple understanding of concepts including color, space,time
                -use “in” and ”on”
                -use of simple pronouns ex;  “me” ,  “you”
                -use of 2 step directions 
                -simple, repetitive, and rhythmic language.
                -Sentences should be short and easy to understand.
                -  The plot should be straightforward with minimal complexity, often focusing on everyday experiences or simple adventures.
                -Incorporating a sing-song quality or actual songs
                -Interactive content
                -Focus on familiar objects, animals, or routines (e.g., bedtime, mealtime) that are part of the child’s everyday life.
                - Simple expressions of emotions can help children begin to understand and identify their own feelings
                Here’s an example of what  i am looking for : 
                Once upon a time, in a sunny meadow, there lived a little bunny named Benny. Benny was a curious and happy bunny who loved hopping around and discovering new things.
                One bright morning, after a rainy night, Benny saw something magical in the sky. It was a beautiful rainbow, stretching across the sky with its lovely colors: red, orange, yellow, green, blue, and purple.
                """
        
    elif ageGroup== "Preschoolers":
        prompt="""  - Simple, relatable narratives
                    - Stories that enhance vocabulary
                    - Themes that encourage critical thinking and empathy
                    -use simple rhymes
                    -use words about time like yesterday, morning or night
                    -use sentences that gives a lot of details
                    -remember that they struggle at the sounds l,s,r,v,z,ch,sh,th. Try to avoid using them
                    -use numbers
                    -touch on subjects like responsibility , pride and guilt
                    -story that comes through actions


                    Here's an example of what im looking for :
                    Once upon a time, in a lush green forest, there lived a little red fox named Felix. Felix
                    was very curious and loved to explore every corner of the forest. One sunny morning, Felix
                    decided to go on an adventure to find the biggest and juiciest berry in the whole forest.
                    Felix hopped along the path, greeting all his friends. He waved to Mr. Owl perched high in
                    the tree and said hello to Mrs. Rabbit who was busy collecting carrots. Everyone loved Felix
                    because he was always kind and helpful.
                """
        
    elif ageGroup== "Early Elementary":
        prompt = """   - Straightforward and clear structure of the plot
            - Predictable beginning, middle, and end
            - Relatable Characters
            - Few main characters
            - Child protagonists or animal characters
            -Simple Language
            - Short, concise sentences
            - Use of repetition
            - Clear, simple message
            - Positive outcomes
            - Predictable patterns
            - Questions to engage the reader
            - Rhyming text
            - Musical, sing-song quality
            Here’s an example of what i am looking for : 
            Once upon a time, in a cozy burrow, lived a brave little bunny named Benny
            -Benny loved to hop around the meadow and play with his friends
            - One sunny day, Benny’s friends were too scared to explore the forest
            - Benny decided to go alone to find new adventures
            - In the forest, Benny found a lost kitten, Mia, who needed help finding her way home 
                """

    elif ageGroup== "Preteens":
        prompt= """- characters who demonstrate empathy, teamwork, and cooperation.
            - Include themes that address and normalize mood swings, emotional changes, and providing relatable scenarios and resolutions.
            -Introduce conflicts that the characters must navigate and resolve, reflecting the quarrels and disagreements common in this age group.
            - Highlight characters who experience and manage sensitive social situations, teaching readers about empathy, kindness, and dealing with hurt feelings.
            - Create plots centered around group activities, such as school projects, sports, or clubs, emphasizing the importance of teamwork and the joy of making new friends.
            -Focus on themes of loyalty, trust, and the complexities of peer relationships, showing characters standing by their friends and working through challenges together.
            -Feature characters who navigate peer pressure and develop their unique identities, even while sometimes conforming to group norms.
            Promoting gender equality and mutual respect.
            -Include elements of secret codes, puzzles, and practical jokes to engage the reader’s sense of fun and adventure.
            - Provide examples of characters dealing with teasing and criticism, demonstrating resilience, self-confidence, and the ability to stand up for themselves and others.
            - Weave in moral lessons and ethical dilemmas, helping readers develop a strong sense of right and wrong through the characters’ experiences and choices.
            -Incorporate elements of hero worship, where characters admire and aspire to be like their heroes, whether they are real people, fictional characters, or historical figures.
            - Ensure the story has an element of adventure, allowing the characters to engage in heroic acts, face challenges, and grow from their experiences.
            Here’s an example of what i am looking for: 
            In the quiet town of Willow Creek, nestled between rolling hills and ancient forests, lived a group of friends who were always ready for an adventure. The leader of the group was Alex, a curious and resourceful twelve-year-old with a knack for solving mysteries. Alongside him were his best friends, Emma, a talented artist with a keen eye for detail, and Jake, a tech whiz who could hack into anything.
            One crisp autumn afternoon, while exploring the outskirts of town, the trio stumbled upon an old, abandoned mansion. The mansion, known as Holloway House, had been the subject of many local legends. Some said it was haunted, while others believed it hid a treasure left behind by its eccentric owner, Professor Holloway, who vanished mysteriously many years ago.
            """

    elif ageGroup== "Late Elementary":
        prompt= """ -asks more complex questions
                    -desires detailed answers
                    -shows unusual interest in numbers
                    -likes active, competititce games
                    -enjoys simple games such as checkers and cards
                    -understands the value of coins
                    -enjoys hobbies and collections
                    -likes to experiment
                    -Understand such terms as: alike, different, beginning, end , etc
                    -enjoys mysteries, adventure stories and biographies
                    -Simple, relatable narratives
                    - Familiar settings and relatable characters
                    - Introduction to problem-solving and critical thinking
                    - Promotes empathy and understanding of diverse perspectives
                    Here's an example of what i am looking for :
                    Once upon a time, in a cozy little burrow nestled in a meadow, lived a bunny named Benny. 
                    Benny was not like the other bunnies; he was curious and loved exploring the world beyond
                    the safety of his home. His fluffy fur was white as snow, and his eyes sparkled with 
                    excitement whenever he embarked on a new adventure.
"""

    else:
        raise ValueError("choose target error")

    return prompt

#this is the function that per age group define the characteristics of the question generation
def questionChar(ageGroup):
    if ageGroup=="Toddlers":
        prompt = """
            -asks questions about event descriptions in the story 
            -ask to provide solutions  to the problem mentioned in the story 
            -”Who” ,”What” and “What happened” questions 
            -straighfoward questions about characters or objects mentioned in the story
            - Questions about the feelings of the characters 
            -Questions related to the future of the story , what can happen next 
            -Questions about the size and colors in the story
            -Ask questions that have short answers
            """
    elif ageGroup== "Preschoolers":
        prompt = """
            "-ask simple questions "
            "-ask questions about time "
            "-ask questions about why and how . for example  “that’s a nice bridge he is  building, Why he put it there”"
            "-ask about his opinions about story"
        """
    elif ageGroup==  "Early Elementary":
        prompt = """ "-ask simple questions "
                "-ask questions about time "
                "-ask questions about why and how . for example  “that’s a nice bridge he is  building, Why he put it there”"
                "-ask about his opinions about story"
            """
    elif ageGroup== "Late Elementary":
        prompt = """-questions about comprehension of the problems, solutions mentioned in the story
                    -questions about character’s behavious
                    -ask about questions that requires reading between the lines
                    - Why questions
                    -ask questions that requires identifying compare and contrast elements
                    -questions that focuses on reasons and outcomes
                    -Opinion based questions , open ended questions
                    -predictive questions about future
                    -questions that mesure understanding of the story
                    """
    elif ageGroup == "Preteens":
        prompt = """
            -Asking questions about how characters deal with problems and how would they deal about it
            -Asking to describe challenges faced
            -Asking the solutions for the problems mentioned in the story
            -Asking opinions about characters
            -Asking if an act is right or wrong in the text
            -Asking what we can learn from character’s reactions
            -Asking about clues, or secret codes that friends used to communicate in the story
            -Questions related to the storyline and empathy
            """
    else: 
        raise ValueError("question characteristics error")
    return prompt

def lectureChar(ageGroup):
    if ageGroup == "Toddlers":
        prompt = """
                    -easy language
                    -simple words
                    -simple phrases
                    -when you use terminology, simplify them by easy definitions that toddlers see in their daily life 
                    and that they can relate or imagine easily
                    -imagine that this is going to be the story where toddlers can imagine in their head 100%"""
    elif ageGroup == "Preschoolers":
        prompt = """ -Stories that enhance vocabulary
                - Themes that encourage critical thinking and empathy
                -use simple rhymes
                -use words about time like yesterday, morning or night
                -use sentences that gives a lot of details
                -rememver that they struggle at the sounds l,s,r,v,z,ch,sh,th. Try to avoid using them 
                -use numbers 
                -touch on subjects like responsibility , pride and guilt
                -story that comes through actions"""
    elif ageGroup== "Early Elementary" : 
        prompt = chooseTarget("Early Elementary")
    elif ageGroup == "Late Elementary": 
        prompt= chooseTarget("Late Elementary")
    elif ageGroup== "Preteens":
        prompt = chooseTarget("Preteens")
        
    else: 
        raise ValueError("lecture char errrorr")
    return prompt
#generate story based on questions ==> used the method few shot prompting
def gSbA(question_segment, age_group, word_count=300, model= MODEL, temperature = 0.7, max_tokens = 1000):
    system_prompt = (f"""Given the following questions : 
                     ```
                     {question_segment}
                     ```
                     Generate a story that would have answer to the questions
                     here is an example of what i'm looking for:
                     Questions:What's the name of Nur's Robot ? How's the weather today? What was Robot doing when Nur woke up
                     Generated Story: Today was a sunny day. Nur opened her eyes to her Robot Alpha-Mini singing a morning song for her.                    
                     For the given age group:
                     ```
                     {age_group}
                     ```
                     under  {word_count} words
                    here is an example of what i'm looking for:
                    {chooseTarget(age_group)}                          
                                    """)

    response = client.chat.completions.create(
        model=model,
        messages= [
            { "role": "system",
             "content" : system_prompt },
        ],       
        n=3,
        temperature=temperature,
        max_tokens=max_tokens,
    ) 

    return response.choices[0].message.content

def regenerateStory(storyPrompt,model=MODEL, temperature=0.8,max_tokens= 1000):
    system_prompt = "You are extremely warm and imaginative assistant, responsible with rewriting tales"
    "for a special audience: 5-year-old children. Your task is to remodel the story "
    "into an creative new version. This upgraded story should be in the style of"
    "babysitter: easygoing, understanble language, engaging, filled with curiosity, and with a fantastic plot twist to"
    "captivate children. Change the ending of the tale to bring curiousity .Based on this, rewrite the following story into"
    "a magical fantastic and andventuorously appropriate for a 5-year-old children:\n\n"
    "Original Story Segment:"

    response= client.chat.completions.create(
        model=model,
        messages=[
            { "role" : "system",
             "content" : system_prompt},
            { "role" : "user",
             "content": storyPrompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

#giving more specific task to make it understand better. If i dont say "give answer and cite" it does generate questions that doesnt have answers in the text 
def generateQuestions(storyPrompt, questionAbout, target,  model=MODEL, temperature =0.8, max_tokens=1000): 
    system_prompt = (f"You are extremely warm and and creative babysitter, ask 3 questions that have short answers about the following story:"
    f"'{storyPrompt}'\n\n"
    "ask question about"
    f"'{questionAbout}'"
    "Here's the characteristics of the target audience: "
    f"'{target}'"
    "If the questions are about the storyline, use the following features:"
    f"'{questionChar(target)}'"
    "and mix the questions in total there should be 3 questions"
    "dont define question type title per question"
    "per question , cite the answer and then write the next question and answer pair"
    "If the questions are about lecture content, ask questions about the lecture"    
    "Give answer to generated questions and cite from the text"
    "First provide Questions seperately under the title 'Questions' then write down answers under the title 'Answers' "
    """Here's an example of what i am expecting as the structure if the language is english :
        Questions:
            1. What did Hello Kitty and Seb find while chasing butterflies?
            2. How did Hello Kitty, Seb, and Luna travel through the forest?
            3. Why did Luna join Hello Kitty and Seb in their meadow games?

        Answers:
            1. Hello Kitty and Seb found a tiny lost kitten named Luna while chasing butterflies.
            2. Hello Kitty, Seb, and Luna traveled through the forest by crossing bridges and streams.
            3. Luna joined Hello Kitty and Seb in their meadow games because they helped her and formed a bond of friendship.

        Here's an example of what i am expecting as the structure if the language is french :
        Questions:
            1. Qu'ont trouvé Hello Kitty et Seb en chassant les papillons ?
            2. Comment Hello Kitty, Seb et Luna ont-ils voyagé à travers la forêt ?
            3. Pourquoi Luna a-t-elle rejoint Hello Kitty et Seb dans leurs jeux de prairie ?
        
        Answers: 
            1. Hello Kitty et Seb ont trouvé un petit chaton perdu nommé Luna en chassant les papillons.
            2. Hello Kitty, Seb et Luna ont traversé la forêt en traversant des ponts et des ruisseaux.
            3. Luna a rejoint Hello Kitty et Seb dans leurs jeux de prairie parce qu'ils l'ont aidée et ont noué un lien d'amitié.

        Here's an example of what i am expecting as the structure if the language is german :
        Questions: 
            1. Was haben Hello Kitty und Seb bei der Schmetterlingsjagd gefunden?
            2. Wie sind Hello Kitty, Seb und Luna durch den Wald gereist?
            3. Warum hat Luna Hello Kitty und Seb bei ihren Wiesenspielen begleitet?

        Answers: 
            1. Hello Kitty und Seb haben bei der Schmetterlingsjagd ein kleines verlorenes Kätzchen namens Luna gefunden.
            2. Hello Kitty, Seb und Luna gingen durch den Wald und überquerten Brücken und Bäche.
            3. Luna begleitete Hello Kitty und Seb bei ihren Wiesenspielen, weil sie ihr halfen und eine Freundschaft knüpften.        
    """)   

    response= client.chat.completions.create(
        model=model,
        messages=[
            { "role" : "system",
             "content" : system_prompt},
        ],
        n=3, #nb of questions
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

#goal: to extract questions generated by "generateQuestions"
def extractQuestion(questionPrompt, model=MODEL, temperature =0.8, max_tokens=1000):
    system_prompt=(f"extract the phrases that end with a question mark in this: "
                   f"'{questionPrompt}'\n\n")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",             
             "content": system_prompt}
        ],
        n=3, 
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
#goal: when openai generated a story with limited amount of words, it  doesnt necessarily end the phrase
#this function aims to end the story
def story_end(story_prompt):
    end_dot= story_prompt.rfind('.') # finds the last referenced index of the given occurence
    end_exc= story_prompt.rfind('!')
    end_ques = story_prompt.rfind('?')

    end_last=end_dot
    if(end_last<end_exc): end_last= end_exc
    if(end_last<end_ques): end_last= end_ques

    return story_prompt[:end_last+1]


#strings for the answer_question function
sys_mes = """You are a humanoid robot named QT whose job is to help a 5 year old student with any question they have. 
You can show facial expressions and move your arms, but you cannot walk. Your goal is to simulate human to human conversation."""
sys_french = """Vous êtes un robot humanoïde nommé QT dont le travail consiste à aider un élève de 5 ans à répondre à toutes ses questions.
Vous pouvez montrer des expressions faciales et bouger vos bras, mais vous ne pouvez pas marcher. Votre objectif est de simuler une conversation d'homme à homme."""

#Obtains an answer to the given question 
def answer_question(prompt, system_message = sys_french, max_tokens = 200, n = 1):
    return generate_response(prompt, system_message, max_tokens, n)

#Translates the given message. Used for pre-programmed messages that QT will say
def translate(message, language_from = 'en', language_to = 'en'):
    language = None
    if not(language_from == language_to):
        if(language_to == 'fr'):
            language = 'French'
        else:
            language = 'German'
        message = generate_response("Translate the following text to " + language + " Dont add anything extra: " + message )
    print(message)
    return message


def translate_questions(message,language_from = 'en', language_to = 'en', model=MODEL, temperature=0.7, max_tokens=1000):
    language = None
    if not(language_from == language_to):
        if(language_to == 'fr'):
            language = 'French'
        else:
            language = 'German'
    system_prompt = (
        f"Translate the following text '{message}' into {language}. "
        "Ensure that the headers 'Questions:' and 'Answers:' remain unchanged and English. Everything else should be in the translated language"
        "Do not add or remove any content beyond the translation of the message body."
        "Example: translate the following to french"
        "Questions: "
        "1. Where are you?"
        "2. What's your name?"
        "Answers: "
        "1. London"
        "2. Nur"
        "translated version is the following: "
        "Questions: "
        "1. Ou es-tu?"
        "2. Comment tu t'appelles?"
        "Answers: "
        "1. Londre"
        "2. Nur"
    )
   
    response = client.chat.completions.create(
        model=model,
        messages= [ {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
   

#debugging purposes
def generate_fake_response(prompt):
    return prompt

#purpose: story generation based on the lecture content 
def generate_lecture_story(story_segment,word_count,ageGroup, model=MODEL, temperature=0.7, max_tokens=1000):
    system_prompt = (
        "You are a extremely creative and friendly assistant, responsible with storifying the lectures"
        "Your task is to remodel the lecture to a story "
        "Here’s the lecture content : "
         f"'{story_segment}'\n\n"
        "Define the subject"
        "Clearly outline the main points and key information of the given prompt "
        "Structure your content with a clear beginning, middle, and end."
        "Identify areas where storytelling elements (characters, setting, conflict, resolution) can be naturally integrated. "
        "Dont list element "
        "Dont use the word 'for beginners' "
        "Here are some characteristics i am expection while generating: "
        f"'{lectureChar(ageGroup)}'" 
        "imagine how would you describe"
         f"'{story_segment}'\n\n"
         "to a "
         f"'{ageGroup}'"         
        "Tone and style : "
        "-Educational yet engaging, with a focus on making complex concepts easy to understand."
        "-Use dialogue and interactions to illustrate key points"
        "under"
        f"'{word_count}'" 
        "words" 
        )  
         
  

    response = client.chat.completions.create(
        model=model,
        messages= [ {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content

#purpose : story generation based on the lecture and lecture's subtopics 
def generate_lecture_subtopics(story_segment,word_count, ageGroup,model=MODEL, temperature=0.7, max_tokens=1000):
    system_prompt = (
        "You are a extremely creative and friendly assistant, responsible with storifying the lectures"
        "Your task is to remodel the lecture to a story "
        "Here’s the lecture content titles : "
        f"'{story_segment}'\n\n"
        "Define the subject."
        "Clearly outline the main subtitles and provide explanation  throughout out the story"
        "Structure your content with a clear beginning, middle, and end."
        "Identify areas where storytelling elements (characters, setting, conflict, resolution) can be naturally integrated. "
        "Dont list element "
        "Dont use the word 'for beginners' "
        "Here are some characteristics i am expection while generating: "
        f"'{lectureChar(ageGroup)}'" 
        "imagine how would you describe"
         f"'{story_segment}'\n\n"
         "to a "
         f"'{ageGroup}'"     
        "Tone and style : "
        "-Educational yet engaging, with a focus on making complex concepts easy to understand."
        "-Use dialogue and interactions to illustrate key points"
        
        "under"
        f"'{word_count}'" 
        "words" )  

    response = client.chat.completions.create(
        model=model,
        messages= [ {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )
 
    return response.choices[0].message.content

#purpose: story generation based on the lecture topic
def generate_lecture_topic(story_segment,word_count,ageGroup, model=MODEL, temperature=0.7, max_tokens=1000):
    system_prompt = (
        "You are a extremely creative and friendly assistant, responsible with storifying the lectures"
        "Your task is to remodel the lecture to a story "
        "Here’s the lecture titles : "
        f"'{story_segment}'\n\n"
        "Define the subject."
        "Define the subheading of the topic for the beginner level "
        "Clearly outline the main subtitles and provide explanation  throughout out the story"
        "Structure your content with a clear beginning, middle, and end."
        "Identify areas where storytelling elements (characters, setting, conflict, resolution) can be naturally integrated. "
        "Dont list element "
        "Dont use 1. 2. notation create a storyline"
        "Dont use the word 'for beginners' "  
        "Here are some characteristics i am expection while generating: "
        f"'{lectureChar(ageGroup)}'"     
        "imagine how would you describe"
         f"'{story_segment}'\n\n"
         "to a "
         f"'{ageGroup}'"             
        "Tone and style : "
        "-Educational yet engaging, with a focus on making complex concepts easy to understand."
        "-Use dialogue and interactions to illustrate key points"
        "under"
        f"'{word_count}'" 
        "words"
    )  

    response = client.chat.completions.create(
        model=model,
        messages= [ {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )
 
    return response.choices[0].message.content

#purpose: story generation based on the lecture topic
def generate_summary(story_segment, model=MODEL, temperature=0.7, max_tokens=1000):
    system_prompt = (
        "summarize the following story: "
        f"'{story_segment}'\n\n"
        "under 1000 characters"
    )  

    response = client.chat.completions.create(
        model=model,
        messages= [ {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )
 
    return response.choices[0].message.content


def return_question_answer(question_prompt,task, model=MODEL, temperature=0.7, max_tokens=1000):
    system_prompt2 = (
         "Here's the questions and answers to it"
         f"'{question_prompt}'\n\n"
        "Do the following task"
         f"'{task}'\n\n"
         """Here's an example of what i am expecting 
         if the question prompt is the following one :
        Questions: 
        1. What color were Lily's wings in the story?
        2. Who did Lily and Lucas meet in the garden?
        3. How did Lili and Lucas feel after their adventure in the garden ?
        Answers: 
        1. Lily had shiny red wings in the story.
        2. Lily and Lucas met a garden gnome in the garden.
        3. Lily and Lucas felt happy and content after their adventure in the garden.
        
        if the task is the following "Return the first question, without any numaration infront " then return for this case "What color were Lily's wings in the story?"
        if the task is the following "Return the second question, without any numaration infront " then return for this case "Who did Lily and Lucas meet in the garden?"
        if the task is the following "Return the third question, without any numaration infront " then return for this case "How did Lili and Lucas feel after their adventure in the garden ?"
        if the task is the following "Return the first answer without any numaration infront"   then return for this case "Lily had shiny red wings in the story."
        if the task is the following "Return the second answer without any numaration infront"  then return for this case " Lily and Lucas met a garden gnome in the garden."
        if the task is the following "Return the third answer without any numaration infront"  then return for this case "Lily and Lucas felt happy and content after their adventure in the garden."   
        -Make sure that the question ends with a question mark , dont take question and answer together, they are asked seperatly     
        """ 
        
        )
    system_prompt3 = (
    "You will be given a question and answer set, and a specific task to perform in any language."
    " Your goal is to follow the task instructions precisely, regardless of the language."
    "\n\n"
    "Question Prompt:\n"
    f"'{question_prompt}'\n\n"
    "Task:\n"
    f"'{task}'\n\n"
    "Instructions:\n"
    "- If the task asks for a specific question, return only that question without any numbering or additional text."
    "- If the task asks for a specific answer, return only that answer without any numbering or additional text."
    "- Ensure that questions end with a question mark (or the equivalent in the given language)."
    "- Do not combine questions and answers in your response."
    "- Handle the task appropriately, regardless of whether the input is in English, French, German, or any other language."
    "\n\n"
    "Examples:\n"
    "- If the task is 'Return the first question', and the question prompt is: "
    "'1. What color were Lily's wings in the story? 2. Who did Lily and Lucas meet in the garden? 3. How did Lily and Lucas feel after their adventure?' "
    "then return 'What color were Lily's wings in the story?'\n"
    "- If the task is 'Return the first answer', and the answer prompt is: "
    "'1. Lily had shiny red wings in the story. 2. Lily and Lucas met a garden gnome in the garden. 3. Lily and Lucas felt happy and content after their adventure in the garden.' "
    "then return 'Lily had shiny red wings in the story.'\n\n"
    "- For French or German, handle the task similarly:\n"
    "- Example for French Question: '1. De quelle couleur étaient les ailes de Lily dans l'histoire ? 2. Qui Lily et Lucas ont-ils rencontré dans le jardin ?' "
    "If the task is 'Return the first question', return 'De quelle couleur étaient les ailes de Lily dans l'histoire ?'\n"
    "- Example for German Answer: '1. Lily hatte glänzende rote Flügel in der Geschichte. 2. Lily und Lucas trafen einen Gartenzwerg im Garten.' "
    "If the task is 'Return the first answer', return 'Lily hatte glänzende rote Flügel in der Geschichte.'"
    )

    system_prompt = (
    "You will be given a set of questions and answers in any language, and a specific task to perform."
    " Your goal is to follow the task instructions precisely, regardless of the input language."
    "\n\n"
    "Question and Answer Set:\n"
    f"'{question_prompt}'\n\n"
    "Task:\n"
    f"'{task}'\n\n"
    "Instructions:\n"
    "- If the task asks for a specific question, return only that question without any numbering or additional text."
    "- If the task asks for a specific answer, return only that answer without any numbering, prefixes, or additional text."
    "- Ensure that answers do not contain any preceding numbers, punctuation, or list indicators like '1.', '2.', etc."
    "- Do not include 'Question' or 'Answer' labels in the response."
    "- Do not combine questions and answers in your response."
    "- Focus on returning only the exact text of the requested question or answer."
    "- This should work equally well for inputs in English, French, German, or any other language."
    "\n\n"
    "Examples:\n"
    "- If the task is 'Return the first answer', and the answer prompt is: "
    "'1. Luna decided to stay in the magical underground world with Orion because she was fascinated by his stories and wanted to learn more about the mystical creatures in that realm.' "
    "then return 'Luna decided to stay in the magical underground world with Orion because she was fascinated by his stories and wanted to learn more about the mystical creatures in that realm.'"
    "\n\n"
    "- For French or German, handle the task similarly:\n"
    "- Example for French Answer: '1. Luna a décidé de rester dans le monde souterrain magique avec Orion car elle était fascinée par ses histoires et voulait en savoir plus sur les créatures mystiques de ce royaume.' "
    "If the task is 'Return the first answer', return 'Luna a décidé de rester dans le monde souterrain magique avec Orion car elle était fascinée par ses histoires et voulait en savoir plus sur les créatures mystiques de ce royaume.'"
    "\n\n"
    "- Example for German Answer: '1. Luna beschloss, in der magischen Untergrundwelt mit Orion zu bleiben, weil sie von seinen Geschichten fasziniert war und mehr über die mystischen Kreaturen dieses Reiches erfahren wollte.' "
    "If the task is 'Return the first answer', return 'Luna beschloss, in der magischen Untergrundwelt mit Orion zu bleiben, weil sie von seinen Geschichten fasziniert war und mehr über die mystischen Kreaturen dieses Reiches erfahren wollte.'"
)


    
  
         
  

    response = client.chat.completions.create(
        model=model,
        messages= [ {   "role" : "system",  "content" : system_prompt },
        ],   
        top_p=1.0,    
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content

#debugging purposes
if __name__ == '__main__':
    prompt = """Questions:
            1. What did Hello Kitty and Seb find while chasing butterflies?
            2. How did Hello Kitty, Seb, and Luna travel through the forest?
            3. Why did Luna join Hello Kitty and Seb in their meadow games?

        Answers:
            1. Hello Kitty and Seb found a tiny lost kitten named Luna while chasing butterflies.
            2. Hello Kitty, Seb, and Luna traveled through the forest by crossing bridges and streams.
            3. Luna joined Hello Kitty and Seb in their meadow games because they helped her and formed a bond of friendship."""
    print(return_question_answer(prompt, "Return the first question"))
    print(return_question_answer(prompt, "Return the second question"))
    print(return_question_answer(prompt, "Return the third question"))
    print(return_question_answer(prompt, "Return the first answer"))
    print(return_question_answer(prompt, "Return the second answer"))
    print(return_question_answer(prompt, "Return the third answer"))

    


   # print(answer_question("Can you dance?")) 
   
    #topic = "Tom and Jerry"
   # toddlers = "Toddlers"
   # preschoolers = "Preschoolers"
    #earlyElementary = "Early Elementary"
   # preTeens = "Preteens"
   # lateElementary = "Late Elementary"
   # lecture= "photosythesis"

    #print("Generation for EARLY ELEMENTARY")
   # story =generate_lecture_topic(lecture, 200, earlyElementary)
    #print(story)
    #print(generateQuestions(story, "storyline", earlyElementary))
   # print(generateQuestions(story, "lecture", earlyElementary))


    
 
