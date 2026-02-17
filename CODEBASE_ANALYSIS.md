# MI2US Codebase Analysis

## Overview

The existing MI2US_Year2s project is a web-based interface for teachers to create AI-powered storytelling activities with social robots. The system uses OpenAI's GPT-3.5 API for content generation and DALL-E for image creation.

---

## Architecture

### 1. Main Application (`main.py`)
**Purpose:** Central controller orchestrating the entire application flow

**Key Components:**
- `StorytellingApp` class - Main application state machine
- State management for greetings, story generation, question/answer flow
- Language configuration (English, French, German)
- Virtual robot interaction handling

**Flow:**
```
User Input → Language Selection → Story Generation → 
Question Generation → Image Generation → Robot Interaction → Evaluation
```

**Key Functions:**
- `greet()` - Initiates session, receives story parameters
- `config_language()` - Sets language (en/fr/de)
- Handles user inputs via WebSocket server

---

### 2. AI Story Generation (`storytelling.py`)
**Purpose:** All LLM interactions for educational content creation

**Current Implementation:** Uses OpenAI GPT-3.5-turbo API

**Key Functions:**

#### Story Generation
- `dSgDaG(topic, age_group, word_count)` - Generate story from topic
- `complete_story(story_segment)` - Continue/expand story
- `generate_lecture_story()` - Convert lecture content to story
- `generate_lecture_topic()` - Generate story from lecture topic
- `generate_lecture_subtopics()` - Story from lecture subtopics

#### Question/Answer Generation
- `generateQuestions(story, questionAbout, target)` - Generate 3 questions with answers
- `gAbQaS(story, questions)` - Generate answers for given questions
- `gSbA(questions, age_group)` - Generate story that answers questions
- `answer_question(prompt)` - Robot Q&A functionality

#### Content Modification
- `mGs(story, indications)` - Modify story based on instructions
- `regenerateStory()` - Create new version of story
- `extractQuestion()` - Extract questions from generated text
- `story_end()` - Ensure story ends properly

#### Translation
- `translate(message, language_from, language_to)` - Translate text
- `translate_questions()` - Translate Q&A pairs (keeps headers in English)

#### Age-Specific Prompting
- `chooseTarget(ageGroup)` - Returns characteristics for age groups:
  - Toddlers (1-3 years)
  - Preschoolers (3-5 years)
  - Early Elementary (5-7 years)
  - Late Elementary (8-10 years)
  - Preteens (11-13 years)

- `questionChar(ageGroup)` - Question generation guidelines per age
- `lectureChar(ageGroup)` - Lecture content characteristics per age

**System Prompts:** Highly detailed with examples for few-shot learning

---

### 3. Image Generation (`imageGeneration.py`)
**Purpose:** DALL-E 3 integration for visual storytelling

**Key Functions:**
- `generate_image(prompt)` - Create image from text prompt
- `generate_image_hint(story, question, answer, name)` - Generate hint images for questions
- `generateImagePrompt(story, age_group)` - Create detailed DALL-E prompts
- `chooseTarget(ageGroup)` - Age-appropriate image style guidelines
- `generate_image_begin()` - Main entry point for image generation

**Image Specifications:**
- Model: DALL-E 3
- Size: 1024x1024
- Format: Base64 JSON
- Output: PNG files (DALLE.png, DALLE1.png, etc.)

**Age-Specific Visual Guidelines:**
- Toddlers: High contrast, simple shapes, primary colors
- Preschoolers: Symbols, interactive elements, bright colors
- Elementary: Action-oriented, story-driven, relatable characters
- Preteens: Complex scenes, visual cues, detailed environments

---

### 4. Server Communication (`server.py`)
**Purpose:** WebSocket server for real-time browser-Python communication

**Architecture:**
- Asynchronous WebSocket server (localhost:10000)
- Thread-based execution
- Client management (multiple connections supported)
- Bidirectional communication (receive user input, send robot feedback)

**Key Functions:**
- `handler(websocket, path, callback)` - Handle incoming WebSocket messages
- `run_server(callback)` - Start WebSocket server in event loop
- `send_data(feedback)` - Broadcast to all connected clients
- `await_response()` - Block until client response received
- `start_thread()` - Run server in separate thread

**Communication Protocol:**
- Receives: User inputs (story prompts, questions, answers)
- Sends: Robot responses, generated content, state updates
- Special command: "close" - Terminates connection and exits

---

## Web Interface

### HTML Files

1. **`index.html`** - Main landing page
2. **`storyGeneration.html`** - Story creation interface
3. **`lectureGeneration.html`** - Lecture content interface
4. **`robotInteraction.html`** - Robot control panel
5. **`virtualRobotInteraction.html`** - Virtual robot simulation
6. **`survey.html`** - Feedback collection
7. **`userGuide.html`** - Documentation

### Frontend Stack
- HTML5 + CSS3 (`styles.css`)
- JavaScript (WebSocket client)
- DALL-E generated images
- Video content (`newCHILI.mp4`)

---

## Data Flow

```
Teacher Input (Web UI)
    ↓
WebSocket → server.py → main.py
    ↓
storytelling.py (OpenAI GPT-3.5)
    ↓
Story/Questions Generated
    ↓
imageGeneration.py (DALL-E 3)
    ↓
Images Saved Locally
    ↓
Content Sent Back via WebSocket
    ↓
Robot Performs Actions
    ↓
Student Interaction
    ↓
Evaluation/Feedback (survey.json, savedSessions.json)
```

---

## Key Observations

### Strengths
1. **Comprehensive age-group targeting** - Detailed prompts for 5 age groups
2. **Multilingual support** - English, French, German
3. **Flexible content generation** - Story from topic, questions, or lecture content
4. **Real-time interaction** - WebSocket enables responsive UI
5. **Visual enhancement** - Automatic image generation

### Limitations
1. **External API dependency** - Requires OpenAI API keys and internet
2. **Cost concerns** - GPT-3.5 + DALL-E usage costs money per request
3. **Alpha Mini specific** - Robot control code tied to one platform
4. **No local model** - Cannot run offline
5. **Limited error handling** - API failures not gracefully handled

---

## Migration Path to Phi-Ed + Reachy Mini

### What Needs to Change

1. **Replace OpenAI API** with local Phi-Ed model
   - Install: `transformers`, `peft`, `torch`
   - Load model: `Mortadha/Phi-Ed-25072024`
   - Implement local inference pipeline
   - Adjust prompt formats for Phi-3

2. **Replace Alpha Mini** with Reachy Mini
   - Install: `reachy_mini` SDK
   - Rewrite robot control functions
   - Map Alpha Mini actions → Reachy Mini equivalents
   - Update hardware interface

3. **Image Generation** (Optional)
   - Keep DALL-E (requires API key) OR
   - Replace with Stable Diffusion (local generation)

4. **Maintain Core Logic**
   - Age-group targeting prompts (reusable!)
   - Story generation flow
   - Question/answer mechanisms
   - WebSocket server architecture
   - Web UI (minor updates only)

---

## API Keys Required (Current System)

Located in:
- `storytelling.py` line 4-6
- `imageGeneration.py` line 13-15

```python
OpenAI.api_key = "YOUR_KEY"
os.environ["OPENAI_API_KEY"] = "YOUR_KEY"
```

**For Migration:** These will be removed when switching to local Phi-Ed model!

---

## Files to Preserve

- Age-group prompts and characteristics
- Question generation logic
- Language translation framework
- WebSocket server architecture
- Web UI design

## Files to Rewrite

- LLM inference calls (OpenAI → Phi-Ed)
- Robot control (Alpha Mini → Reachy Mini)
- Image generation (optional: DALL-E → Stable Diffusion)

---

## Dependencies (`requirements.txt`)

Current dependencies (note the strange encoding - file needs fixing):

```
annotated-types==0.7.0
anyio==4.4.0
certifi==2024.7.4
charset-normalizer==3.3.2
colorama==0.4.6
distro==1.9.0
h11==0.14.0
httpcore==1.0.5
httpx==0.27.0
idna==3.8
jiter==0.5.0
openai==1.42.0
pillow==10.4.0
pydantic==2.8.2
pydantic_core==2.20.1
requests==2.32.3
sniffio==1.3.1
tqdm==4.66.5
typing_extensions==4.12.2
urllib3==2.2.2
websockets==13.0
```

**For Migration, ADD:**
```
transformers>=4.40.0
peft>=0.10.0
torch>=2.1.0
accelerate>=0.27.0
bitsandbytes>=0.43.0  # For QLoRA
reachy-mini>=1.0.0  # Reachy Mini SDK
```

---

## Session Data

- `savedSessions.json` - Stores past interactions
- `survey.json` - Feedback responses
- System logs in `Experiment/Generated Output (System Log).txt`

---

## Next Steps

1. Set up local Phi-Ed model inference
2. Test Phi-Ed with existing prompt templates
3. Install Reachy Mini SDK and test basic movements
4. Create adapter layer: old functions → new implementations
5. Test end-to-end with virtual robot first
6. Deploy to physical Reachy Mini

---

*Document created: February 10, 2026*
