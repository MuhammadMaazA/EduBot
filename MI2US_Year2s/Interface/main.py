#for the virtual robot part 
import asyncio
import json
import os
from pathlib import Path
import webbrowser
import re
import storytelling as ai
import imageGeneration as ig
import server
from server import send_data, start_thread,join_thread
from multiprocessing.connection import Client
import tracemalloc

tracemalloc.start()

# For web module
local_data = None

# For evaluation state
done_questions = True
questions = []

chosen_language = -1
language = ''
next_global_state = ''

def check_first_word(text, word):
    words = text.split()
    if word:
        return words[0] == word
    return False

def config_language():
    global chosen_language
    global language

    chosen_language = int(server.await_response())
    if chosen_language == 2:
        language = 'de'
    elif chosen_language == 1:
        language = 'fr'
    elif chosen_language == 0:
        language = 'en'
    print("Language chosen: ", language)
    print()

class StorytellingApp:

    def __init__(self):
        self.story_prompt = ""
        self.selected_questions = ""
        self.suggestions = ""
        self.level = ""
        self.type= ""
        self.sl = 0
        self.question1 = ""
        self.question2 = ""
        self.question3 = ""
        self.answer1 = ""
        self.answer2 = ""
        self.answer3 = ""
        self.image_generate = False
        self.language = " "

    def greet(self):   
       
        # TODO sometimes virtual robot doesnt say some words
        print("Executing state GREETINGS")
        global local_data
        local_data = server.await_response()
        print("LOCAL DATA RECEIVED IN GREETINGS: ", local_data)

        if local_data == 'close':
            self.goodbye()
            return

        inputs = local_data.split("|")
        ai_level = int(inputs[0])
        story = inputs[1]
        self.sl = int(inputs[2]) #how long the text is but here just putting the automatic value of the thing if the value is not updated
        level = inputs[3] #level on age 

        if language == 'de':
            self.language = "german"
        elif language == 'fr':
            self.language = "french"
        else :
            self.language = "english"

        if ai_level == 0:
            self.type = "storyline"
            self.level = level
            self.story_prompt = story
        elif ai_level == 1:
            self.type = "storyline"
            self.level = level
            self.story_prompt = ai.dSgDaG(story, level, str(self.sl))
            if language == 'fr':
                self.story_prompt = ai.translate(self.story_prompt, 'en', 'fr')
            elif language == 'de':
                self.story_prompt = ai.translate(self.story_prompt, 'en', 'de')
        elif ai_level == 2:  
            self.level = level
            self.type = "storyline"
            self.story_prompt = ai.dSgDaG(story, level, str(self.sl), temperature=1.2)
            if language == 'fr':
                self.story_prompt = ai.translate(self.story_prompt, 'en', 'fr')
            elif language == 'de':
                self.story_prompt = ai.translate(self.story_prompt, 'en', 'de')
        elif ai_level == 3:
            self.level = level
            self.type = "storyline"
            if language == 'fr':
                self.story_prompt = ai.complete_story(ai.translate("Gardez l'histoire originale, et tissez la suite de l'histoire à partir d'ici:", language_to=language) + story + "under" + str(self.sl) + "words")
            elif language != 'en':
                self.story_prompt = ai.complete_story(story + "Complete the rest in German under " + str(self.sl) + " words")
            else:
                self.story_prompt = ai.complete_story(story + "Complete the rest of the story in English under " + str(self.sl) + " words")
        elif ai_level == 4:
            self.level = level 
            self.type = "storyline"
            if language == 'fr':
                self.story_prompt = ai.translate(ai.gSbA(story, level, str(self.sl)), 'en', 'fr')
            elif language != 'en':
                self.story_prompt = ai.translate(ai.gSbA(story, level, str(self.sl)), 'en', 'de')
            else:
                self.story_prompt = ai.gSbA(story, level, str(self.sl))
        elif ai_level == 5:
            self.level = level
            self.type = "lecture content"
            if language == 'fr':
                self.story_prompt = ai.translate(ai.generate_lecture_story(story, str(self.sl), level), 'en', 'fr')
            elif language != 'en':
                self.story_prompt = ai.translate(ai.generate_lecture_story(story, str(self.sl), level), 'en', 'de')
            else:
                self.story_prompt = ai.generate_lecture_story(story, str(self.sl), level)
        elif ai_level == 6:
            self.type = "lecture content"
            self.level = level
            if language == 'fr':
                self.story_prompt = ai.translate(ai.generate_lecture_subtopics(story, str(self.sl), level), 'en', 'fr')
            elif language != 'en':
                self.story_prompt = ai.translate(ai.generate_lecture_subtopics(story, str(self.sl), level), 'en', 'de')
            else:
                self.story_prompt = ai.generate_lecture_subtopics(story, str(self.sl), level)
        elif ai_level == 7:
            self.level = level
            self.type = "lecture content"
            if language == 'fr':
                self.story_prompt = ai.translate(ai.generate_lecture_topic(story, str(self.sl), level), 'en', 'fr')
            elif language != 'en':
                self.story_prompt = ai.translate(ai.generate_lecture_topic(story, str(self.sl), level), 'en', 'de')
            else:
                self.story_prompt = ai.generate_lecture_topic(story, str(self.sl), level)

        print(self.story_prompt)
        print("THIS IS GENERATED IN GREETINGS")
        print()

        self.send_story_prompt()
    

    def send_story_prompt(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_data(self.story_prompt))

        self.client_feedback()

    def client_feedback(self):
        global local_data
        print("Executing state CLIENTFEEDBACK")

        local_data = server.await_response()
        print("LOCAL DATA RECEIVED IN CLIENTFEEDBACK: ", local_data)

        if local_data.split(' ', 1)[0] == 'keepStory':
            self.story_prompt =local_data.split(' ', 1)[1]
            self.keep_story()
        elif local_data.split(' ', 1)[0] == 'saveStory':
            self.keep_story()
        elif local_data.split(' ', 1)[0] == 'suggestions':
            self.suggested_story(local_data)
        elif local_data.split(' ', 1)[0] == 'regenerate':
            self.story_prompt =local_data.split(' ', 1)[1] 
            self.regenerate_story()
        else : 
            self.story_prompt = local_data
            self.keep_story()

    def keep_story(self):
        global local_data
        print("Executing state KEEPSTORY")
        local_data = server.await_response()
        print("LOCAL DATA RECEIVED FOR KEEP STORY STATE: ", local_data)      
        if local_data == '1': 
            #this is the state transition if i dont wanna generate image
            self.image_generate = False
            self.goodbye()
            #self.image_generation_state()
        elif local_data == '0':
            self.image_generate = True
            self.query_generation()

    def query_generation(self):
        global local_data
        print("Executing state QUERYGENERATION")
        local_data = server.await_response()
        print("LOCAL DATA RECEIVED FOR QUERY GENERATION STATE: ", local_data)

        if local_data == 'option0is chosen': #human written questions
            self.query_generation_0() 
        elif local_data == 'option1is chosen': #ai written questions
            self.query_generation_1()

    def query_generation_0(self):
        global local_data
        print("Executing state QUERYGENERATION_0")
        local_data = server.await_response()
        print("LOCAL DATA RECEIVED FOR QUERY GENERATION 0 STATE: ", local_data)
        self.selected_questions = local_data

        print("QUESTION 1: ")
        self.question1 = ai.return_question_answer(self.selected_questions, "Return the first question. Remove any numbering, punctuation, or additional formatting ") #SOMETIMES DOES IT WRONG
        print(self.question1)
        print("ANSWER 1")
        self.answer1 = ai.return_question_answer(self.selected_questions, "Return only the first answer . Remove any numbering, punctuation, or additional formatting. ")
        print(self.answer1)
        print("QUESTION 2")
        self.question2 = ai.return_question_answer(self.selected_questions, "Return only the second question. Remove any numbering, punctuation, or additional formatting")
        print(self.question2)
        print("ANSWER 2")
        self.answer2 = ai.return_question_answer(self.selected_questions, "Return only the second answer. Remove any numbering, punctuation, or additional formatting.")
        print(self.answer2)
        print("QUESTION 3")
        self.question3 = ai.return_question_answer(self.selected_questions, "Return only the third question. Remove any numbering, punctuation, or additional formatting.")
        print(self.question3)
        print("ANSWER 3")
        self.answer3 = ai.return_question_answer(self.selected_questions, "Return only the third answer. Remove any numbering, punctuation, or additional formatting.")
        print(self.answer3)

        
        self.selected_questions = "Questions: " +"\n" + "1. " + self.question1 + "\n" + "2. " + self.question2 + "\n" + "3. " + self.question3 +"\n" + "Answers: " + "\n" + "1. " + self.answer1 + "\n" + "2. " + self.answer2 + "\n" + "3. " + self.answer3 
        self.goodbye()

    
    

    def query_generation_1(self, task= " "): 
        global local_data
        print("Executing state QUERYGENERATION_1")
        
      
        if language == 'fr':
                if(task == "regenerate"): 
                    self.selected_questions = ai.generateQuestions(self.story_prompt + "generate questions in french different than the following: " + self.selected_questions, self.type, self.level)
                else: 
                    self.selected_questions = ai.generateQuestions(self.story_prompt + "generate questions in french", self.type, self.level)
                print("Generated questions [french]: ")
                print(self.selected_questions)
        elif language != 'en':
                if(task == "regenerate"): 
                    self.selected_questions = ai.generateQuestions(self.story_prompt + "generate questions in german different than the following: " + self.selected_questions, self.type, self.level)
                else: 
                    self.selected_questions =  ai.generateQuestions(self.story_prompt + "generate questions in german", self.type, self.level)
                print("Generated questions [german]: ")
                print(self.selected_questions)
        else:
            if(task == "regenerate"): 
                    self.selected_questions = ai.generateQuestions(self.story_prompt + "generate questions in english different than the following: " + self.selected_questions, self.type, self.level)
            else: 
                self.selected_questions = ai.generateQuestions(self.story_prompt, self.type, self.level)
                print("Generated questions [english]: ")
                print(self.selected_questions)

        
        self.question1 = ai.return_question_answer(self.selected_questions, "Return the first question, without any numaration infront ")
        self.answer1 = ai.return_question_answer(self.selected_questions, "Return the first answer without any numaration infront ")
        self.question2 = ai.return_question_answer(self.selected_questions, "Return the second question without any numaration infront")
        self.answer2 = ai.return_question_answer(self.selected_questions, "Return the second answer without any numaration infront")
        self.question3 = ai.return_question_answer(self.selected_questions, "Return the third question without any numaration infront")
        self.answer3 = ai.return_question_answer(self.selected_questions, "Return the third answer without any numaration infront")

        print("QUESTION 1: ")
        print(self.question1)
        print("ANSWER 1")
        print(self.answer1)
        print("QUESTION 2")
        print(self.question2)
        print("ANSWER 2")
        print(self.answer2)
        print("QUESTION 3")
        print(self.question3)
        print("ANSWER 3")
        print(self.answer3)

        
        
        #self.selected_questions = "Questions: " +"\n" + "1. " + self.question1 + "\n" + "2. " + self.question2 + "\n" + "3. " + self.question3 +"\n" + "Answers: " + "\n" + "1. " + self.answer1 + "\n" + "2. " + self.answer2 + "\n" + "3. " + self.answer3 
        self.send_questions()

    def send_questions(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_data(self.selected_questions))

        self.query_generation_1_manager()

    def query_generation_1_manager(self):
        global local_data
        print("Executing state QUERYGENERATION1MANAGER")
        local_data = server.await_response()
        questions = re.findall(r'[^.?]*\?', self.selected_questions)
        clean_questions = '\n'.join(question.strip() for question in questions)
        questions_extracted =clean_questions

        if local_data == 'keepQuestions':
            self.selected_questions = questions_extracted
            self.goodbye()
        elif check_first_word(local_data, "Modified"):
            self.selected_questions = local_data
            self.goodbye()
        elif check_first_word(local_data, "Regenerate"):
            self.query_generation_1("regenerate")

    def extract_questions(self, selected_questions, data):
        # Extracting selected questions from the data
        question_numbers = re.findall(r'\d+', data)
        question_list = selected_questions.split('\n')
        selected_list = [question_list[int(num) - 1] for num in question_numbers]
        return '\n'.join(selected_list)
    
    def suggested_story(self, local_data):
        print("Executing state SUGGESTEDSTORY")
        suggestions = local_data.split(' ', 1)[1]
        self.story_prompt = ai.mGs(self.story_prompt, suggestions)
        self.send_story_prompt()

    def regenerate_story(self):
        print("Executing state REGENERATESTORY")
        self.story_prompt = ai.regenerateStory(self.story_prompt + " under " + str(self.sl) + " words" + "in" + self.language + "under" + str(self.sl))
        print(self.story_prompt)
        self.send_story_prompt()

   

    def goodbye(self):
        global local_data
        print("Executing state GOODBYE")
        local_data = server.await_response()
        print("LOCAL DATA RECEIVED in the GOODBYE STATE: ", local_data)
        if(local_data.split(' ', 1)[0]== "robot" ):
            self.robot_interaction()
        elif(local_data=="exit"):
            self.finish_state()
        elif(local_data=="startagain"):
            self.greet()
        else: 
            try:
                dictionary = json.loads(local_data)
                filename= 'C:/Users/NUR/nur_sm/index_files/sunum/savedSessions.json'
                if os.path.exists(filename):
                    with open(filename, 'r') as file:
                        try:
                            data = json.load(file)
                            if isinstance(data, list):
                                data.append(dictionary)
                            else:
                                data = [data, dictionary]
                        except json.JSONDecodeError:
                            data = [dictionary]
                else:
                    data = [dictionary]

                with open(filename, 'w') as file:
                    json.dump(data, file, indent=4)
                #keep the story executed so generate story image based on it here and save it

            except json.JSONDecodeError:
                print("JSONDecodeError: Invalid data received:", local_data)
                # Log the error and proceed without saving if JSON decoding fails

                self.goodbye() # looping back in state but now; saving the entry is not possible
    

    def robot_interaction(self): 
        print("Executing state ROBOT INTERACTION")
        if(self.image_generate== False):
            prompt_image = ig.generate_image_begin(self.story_prompt, self.level)     
            cwd = os.getcwd()    
            print("current working directory", cwd)
            asyncio.run(send_data("allow"))
        if(self.image_generate== True):
            summary = ai.generate_summary(self.story_prompt)
            prompt_image = ig.generate_image_begin(summary, self.level) 
            first_image = ig.generate_image_hint(summary,self.question1, self.answer1, 'DALLE1.png')
            second_image = ig.generate_image_hint(summary,self.question2, self.answer2, 'DALLE2.png')
            third_image = ig.generate_image_hint(summary,self.question3, self.answer3, 'DALLE3.png')
            asyncio.run(send_data("allow"))

        
        global local_data
        local_data = server.await_response()
        print("LOCAL DATA RECEIVED in the ROBOT INTERACTION STATE: ", local_data)
        if(local_data.split(' ', 1)[0]=="exit"):         
           self.finish_state()             

   
    def finish_state(self):
        print("Executing state FINISH")
        global local_data
        local_data = server.await_response()
        print("LOCAL DATA RECEIVED in the FINISH STATE: ", local_data)

        dictionary = json.loads(local_data)
        filename = 'survey.json'

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):
                        data.append(dictionary)
                    else:
                        data = [data, dictionary]
                except json.JSONDecodeError:
                    data = [dictionary]
        else:
            data = [dictionary]

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

        print("State machine execution finished")


def main():
    app = StorytellingApp()
    print("Starting the server")
    server.start_thread()  
    #it is expected form users to put this after their home path to storytellerRobot directory
    home_path = Path.home() / 'storytellerRobot' #e ex; C:\Users\Nur
    index_html_path= home_path/ 'secondary.html'
    
    print("Started the server") 
    print("Opening:", index_html_path)
    webbrowser.open_new_tab(index_html_path)
    config_language()
    print("configured the language")
    print("Starting the server")
  
    app.greet()

    server.join_thread()
    print("Server joined the thread")

if __name__ == '__main__':
    print("STARTING QT MODULE")
    main()
 