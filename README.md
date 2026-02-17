# Educational Robot Project - Reachy Mini with LLM

**Team Members:** You & Yash  
**Supervisor:** Dr. Daniel Tozadore (UCL)  
**Next Meeting:** Thursday (same time & place)

---

## Project Overview

Building an **AI-powered educational robot** using the **Reachy Mini robot** and a fine-tuned **Large Language Model (LLM)** for interactive storytelling and multicultural integration in classrooms.

### Key Components:
- **Robot:** Reachy Mini by Pollen Robotics ([docs](https://huggingface.co/docs/reachy_mini/index))
- **AI Model:** Phi-Ed - Fine-tuned educational LLM ([Hugging Face](https://huggingface.co/Mortadha/Phi-Ed-25072024))
- **Previous Project:** MI2US_Year2s (Alpha Mini robot with LLM storytelling)
- **Purpose:** Interactive storytelling for education with personality-adaptive robot behavior

---

## Resources Provided

### Research Papers (Not for circulation - unpublished):
**Note:** The research papers are not included in this repository as they are not officially published yet. They were shared privately by the professor.

1. `Automated_Robot_Personality_Traits_in_Educational_HRI.pdf` (excluded from repo)
   - Focus: LLM personality traits (extraversion) impact on student learning
   - Method: QLoRA fine-tuning for pedagogical strategies
   - Key findings: Personality affects engagement and learning outcomes

2. `ROMAN_25__RobboBuddy_in_the_Classroom.pdf` (excluded from repo)
   - Focus: Teacher interface for LLM-powered robot storytelling
   - 1-week classroom study with 27 students
   - Framework for scenario-based learning activities

### Code Repository:
- **GitHub:** [MI2US_Year2s](https://github.com/dtozadore/MI2US_Year2s)
- **Local:** `MI2US_Year2s/` folder (previous implementation with Alpha Mini)

### Tools & Links:
- PyQt Designer: https://www.pythonguis.com/installation/install-qt-designer-standalone/

---

## TODO List

### Phase 1: Understanding & Setup (Week 1)

- [x] Receive project resources from professor
- [x] **Task 1:** Review both research papers thoroughly
  - [x] Understand system architecture
  - [x] Study fine-tuning approach (QLoRA)
  - [x] Analyze study methodology and results
- [x] **Task 2:** Explore existing MI2US_Year2s codebase
  - [x] Review `main.py` - main application flow
  - [x] Review `storytelling.py` - AI story generation
  - [x] Review `imageGeneration.py` - visual content creation
  - [x] Review `server.py` - communication layer
  - [x] Understand the web interface (HTML/CSS/JS files)
- [ ] **Task 3:** Set up development environment
  - [ ] Create Python virtual environment
  - [ ] Install dependencies from `requirements.txt`
  - [ ] Test existing code (if possible)

---

### Phase 2: Research & Learning (Week 1-2)

- [x] **Task 4:** Study Reachy Mini robot
  - [x] Read documentation thoroughly
  - [x] Understand hardware capabilities (sensors, actuators)
  - [x] Learn API for movement, speech, and interaction
  - [x] Compare with Alpha Mini (what's different?)
  
- [x] **Task 5:** Explore Phi-Ed model on Hugging Face
  - [x] Review model card and documentation
  - [x] Check model size, requirements, and capabilities
  - [x] Understand input/output format
  - [ ] Test model inference (if possible)

- [ ] **Task 6:** Team coordination with Yash
  - [ ] Schedule meeting to discuss project
  - [ ] Divide responsibilities:
    - Backend/AI work (LLM, robot integration)
    - Frontend/Interface (PyQt Designer, UI/UX)
  - [ ] Set up shared workspace/Git repo
  - [ ] Define communication workflow

---

### Phase 3: LLM Fine-tuning (Week 2-4)

- [ ] **Task 7:** Set up Hugging Face & download Phi-Ed
  - [ ] Create/verify Hugging Face account
  - [ ] Install required libraries (`transformers`, `peft`, `bitsandbytes`)
  - [ ] Download Phi-Ed base model locally
  - [ ] Test basic model inference

- [x] **Task 8:** Learn QLoRA fine-tuning technique
  - [x] Study QLoRA (Quantized Low-Rank Adapters) theory
  - [x] Review paper's fine-tuning methodology
  - [x] Understand personality trait prompting
  - [x] Learn about pedagogical strategy generation

- [ ] **Task 9:** Prepare training dataset
  - [ ] Educational dialogues and conversations
  - [ ] Question-answer pairs for different subjects
  - [ ] Storytelling examples with various personalities
  - [ ] Multicultural content (English, French, German?)
  - [ ] Format data for training

- [ ] **Task 10:** Fine-tune Phi-Ed model
  - [ ] Set up training pipeline
  - [ ] Configure QLoRA parameters
  - [ ] Train adapter for pedagogical strategies
  - [ ] Train adapter for personality-aligned responses
  - [ ] Monitor training metrics

- [ ] **Task 11:** Test fine-tuned model
  - [ ] Validate personality trait expression (extraversion/introversion)
  - [ ] Check educational content quality
  - [ ] Test response appropriateness for children
  - [ ] Evaluate multilingual capabilities
  - [ ] Run benchmark evaluations

---

### Phase 4: Integration (Week 4-6)

- [ ] **Task 12:** Replace Alpha Mini with Reachy Mini
  - [ ] Identify robot control code in existing system
  - [ ] Implement Reachy Mini API calls
  - [ ] Map Alpha Mini functions to Reachy Mini equivalents
  - [ ] Test basic robot movements and speech

- [ ] **Task 13:** Adapt backend for local Phi-Ed model
  - [ ] Replace OpenAI API calls with local model inference
  - [ ] Optimize model loading and response time
  - [ ] Implement prompt engineering for personalities
  - [ ] Handle API key removal (no more external APIs)

- [ ] **Task 14:** Coordinate with Yash on PyQt interface
  - [ ] Design teacher interface mockups
  - [ ] Implement story prompt input
  - [ ] Add question generation controls
  - [ ] Create session management UI
  - [ ] Integrate backend with frontend
  - [ ] Test user workflows

---

### Phase 5: Testing & Deployment (Week 6-7)

- [ ] **Task 15:** End-to-end system testing
  - [ ] Test: Interface → LLM → Robot flow
  - [ ] Test story generation in multiple languages
  - [ ] Test personality trait switching
  - [ ] Test question generation and answering
  - [ ] Performance testing (response time, accuracy)
  - [ ] Bug fixes and optimization

- [x] **Task 16:** Create documentation
  - [x] Installation guide (SETUP_GUIDE.md)
  - [x] User manual for teachers
  - [x] API documentation (REACHY_MINI_GUIDE.md)
  - [x] Code documentation (CODEBASE_ANALYSIS.md)
  - [x] Troubleshooting guide (in SETUP_GUIDE.md)

- [x] **Task 17:** Prepare for next professor meeting
  - [x] Demo current progress (PROGRESS_REPORT.md)
  - [x] Prepare questions and blockers
  - [x] Document findings and challenges
  - [x] Get feedback and next steps

---

## Technical Stack

### Hardware:
- Reachy Mini robot (Pollen Robotics)
- Development laptop with GPU (for model fine-tuning)

### Software:
- **Python 3.x** - Main programming language
- **PyQt** - Desktop interface development
- **Hugging Face Transformers** - LLM inference
- **PEFT (QLoRA)** - Model fine-tuning
- **HTML/CSS/JavaScript** - Web interface components
- **WebSockets** - Real-time communication

### Key Libraries:
```
- openai (current, to be replaced)
- pillow (image processing)
- requests
- websockets
- annotated-types
- pydantic
```

---

## Project Structure

```
Outreach Project/
├── README.md (this file)
├── Resources.md
├── Automated_Robot_Personality_Traits_in_Educational_HRI.pdf
├── ROMAN_25__RobboBuddy_in_the_Classroom.pdf
└── MI2US_Year2s/
    ├── README.md
    ├── Interface/
    │   ├── main.py (main application)
    │   ├── storytelling.py (AI generation)
    │   ├── imageGeneration.py (image creation)
    │   ├── server.py (communication)
    │   ├── requirements.txt
    │   ├── index.html
    │   ├── robotInteraction.html
    │   └── ... (other web files)
    ├── Experiment/ (study data and results)
    └── PhinetunEd/ (fine-tuning resources?)
```

---

## Key Concepts to Understand

### 1. **QLoRA (Quantized Low-Rank Adapters)**
- Efficient fine-tuning method for large language models
- Reduces memory requirements while maintaining performance
- Allows training on consumer GPUs

### 2. **Personality Traits in LLMs**
- Big Five personality dimensions (focus: **Extraversion**)
- Prompt-based personality shaping
- Linguistic patterns and LIWC analysis

### 3. **Scenario-Based Learning (SBL)**
- Storytelling, role-playing, imagination games
- Engages students through narrative
- Safe environment for problem-solving practice

### 4. **Multicultural Integration**
- Content in multiple languages
- Culturally sensitive scenarios
- Inclusive educational activities

---

## Notes & Questions for Professor

### Questions to Ask:
1. Do we need to support the same languages (English, French, German)?
2. What's the target age group for students?
3. Are there specific educational topics to focus on?
4. How will we test without access to a classroom initially?
5. Budget/resources for GPU compute for fine-tuning?
6. Access to Reachy Mini robot - when and where?

### Challenges to Discuss:
- Fine-tuning compute requirements
- Dataset preparation for training
- Testing methodology without students
- Timeline and milestones

---

## Getting Started (This Week)

### Priority Tasks:
1. Read both research papers
2. Explore the existing codebase
3. Set up Python environment
4. Study Reachy Mini documentation
5. Meet with Yash to divide work

### Quick Commands:
```bash
# Set up virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Mac/Linux

# Install dependencies
cd MI2US_Year2s/Interface
pip install -r requirements.txt

# Run existing code (after configuration)
python main.py
```

---

## Contact

**Professor:** Daniel Tozadore  
**Email:** d.tozadore@ucl.ac.uk  
**Affiliation:** Dept. of Computer Science, University College London (UCL)

**Meeting:** Thursday (recurring, same time & place)

---

*Last Updated: February 10, 2026*
