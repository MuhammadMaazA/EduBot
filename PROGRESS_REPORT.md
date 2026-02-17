# Progress Report for Professor Tozadore

**Student:** Muhammad Maaz  
**Team Member:** Yash  
**Date:** February 10, 2026  
**Next Meeting:** Thursday

---

## Summary

Since receiving the project materials, I have completed an initial analysis of the codebase, reviewed the research papers, and researched the new robot platform (Reachy Mini) and fine-tuning techniques (QLoRA). This report summarizes what I've learned and outlines the path forward.

---

## Completed Tasks

### 1. Project Setup & Organization
- [x] Created project workspace
- [x] Organized files and resources
- [x] Removed unpublished papers from GitHub repository (security)
- [x] Created comprehensive README with project overview and TODO list
- [x] Set up .gitignore to prevent sensitive files from being committed

### 2. Codebase Analysis
- [x] Read and analyzed all Python files:
  - `main.py` - Application flow and state machine
  - `storytelling.py` - LLM integration (850+ lines of prompting logic)
  - `imageGeneration.py` - DALL-E image creation
  - `server.py` - WebSocket communication
- [x] Understood web interface architecture (HTML/CSS/JS)
- [x] Identified dependencies and API requirements
- [x] Documented data flow and system architecture
- [x] **Created detailed CODEBASE_ANALYSIS.md document**

**Key Findings:**
- System currently uses OpenAI GPT-3.5 + DALL-E (requires API keys)
- Excellent age-group specific prompting (Toddlers → Preteens)
- Multilingual support (English, French, German)
- WebSocket-based real-time communication
- Well-structured state machine for interactions

### 3. Reachy Mini Research
- [x] Read official documentation (https://huggingface.co/docs/reachy_mini/)
- [x] Studied hardware specifications
- [x] Learned Python SDK basics
- [x] Explored app ecosystem on Hugging Face
- [x] Compared with Alpha Mini robot
- [x] **Created REACHY_MINI_GUIDE.md document**

**Key Findings:**
- Open-source platform with active community
- Three versions: Wireless (autonomous), Lite (development), Simulation
- App store integration via Hugging Face Spaces
- Strong educational storytelling app potential
- Assembly takes 2-3 hours

### 4. LLM Fine-tuning Research
- [x] Researched QLoRA technique
- [x] Read 2026 tutorials and guides
- [x] Understood memory requirements and benefits
- [x] Studied PEFT library and bitsandbytes
- [x] Reviewed Phi-Ed model on Hugging Face
- [x] **Created QLORA_GUIDE.md document**

**Key Findings:**
- QLoRA enables fine-tuning on consumer GPUs (24GB)
- Phi-Ed is already fine-tuned from Phi-3-Mini
- Can create task-specific adapters without retraining base model
- Training takes hours instead of days
- 99.94% of model stays frozen, only 0.06% trainable

### 5. Research Papers Review
- [x] Started reading "Automated Robot Personality Traits in Educational HRI"
- [x] Started reading "RoboBuddy in the Classroom"
- [x] Understood previous study methodology (38 students, 2 sessions)
- [x] Learned about personality shaping via prompts
- [x] Noted LIWC analysis for linguistic validation

**Key Insights:**
- Extraversion significantly impacts student engagement
- Personality traits can be consistently expressed through language
- Students' prior knowledge affects perception of personality changes
- Scenario-based learning (storytelling) shows high engagement

---

## Documentation Created

I've created 5 comprehensive documents for the project:

1. **README.md** (307 lines)
   - Complete project overview
   - 17-task TODO list organized in 5 phases
   - Technical stack and resources
   - Quick start guide

2. **CODEBASE_ANALYSIS.md**
   - Detailed breakdown of all code files
   - Function-by-function documentation
   - Data flow diagrams
   - Migration strategy from OpenAI → Phi-Ed
   - Migration strategy from Alpha Mini → Reachy Mini

3. **REACHY_MINI_GUIDE.md**
   - Hardware specifications
   - Software API overview
   - Code examples
   - App ecosystem explanation
   - Development workflow recommendations

4. **QLORA_GUIDE.md**
   - QLoRA technique explanation
   - Technical setup instructions
   - Complete code templates
   - Training data preparation guide
   - Troubleshooting tips

5. **PROGRESS_REPORT.md** (this document)
   - Summary of work completed
   - Key findings and insights
   - Questions for discussion
   - Recommended next steps

---

## Key Insights & Observations

### Strengths of Existing System
1. Comprehensive age-group targeting (5 categories with detailed prompts)
2. Strong pedagogical foundation (question generation, story adaptation)
3. Multilingual capabilities
4. Real-time interaction via WebSockets
5. Automatic image generation for visual engagement

### Challenges Identified
1. **External Dependencies:** Current system requires OpenAI API (cost + internet)
2. **Robot Platform Change:** Need to rewrite Alpha Mini code for Reachy Mini
3. **Hardware Access:** Unclear if we have Reachy Mini yet (simulation available)
4. **Training Data:** Need to prepare datasets if additional fine-tuning required
5. **Testing Environment:** Need strategy for testing without students initially

### Opportunities
1. **Local Model:** Phi-Ed runs locally (no API costs, works offline)
2. **Open Source Robot:** Reachy Mini has growing community and app store
3. **Reusable Prompts:** Age-group prompts can transfer to Phi-Ed
4. **Hugging Face Integration:** Can publish our app for wider impact
5. **Extensibility:** QLoRA allows creating specialized adapters

---

## Questions for Thursday's Meeting

### Technical Questions
1. **Phi-Ed Model & Fine-tuning:**
   - Should we use Phi-Ed as-is or fine-tune further?
   - **If we need to fine-tune: What dataset should we use?** Can we access the original training data?
   - Are there specific new capabilities you want us to add?
   - Should we create our own dataset from the existing MI2US prompts and examples?

2. **Reachy Mini Robot:**
   - Do we have access to Reachy Mini hardware?
   - Which version: Wireless, Lite, or start with Simulation?
   - When can we begin physical testing?
   - Where is the robot located (lab space)?

3. **Development Environment & GPU Resources:**
   - **What GPU resources are available at UCL for model training/inference?**
   - Can we use university compute for fine-tuning if needed?
   - Is there a development server we should use?
   - What are the GPU specs (VRAM, CUDA version)?

4. **Testing & Validation:**
   - How do we test without students initially?
   - Do we need ethics approval for user studies?
   - What's the timeline for classroom deployment?

### Project Scope Questions
1. **Languages:** Must we support all three languages (English, French, German)? Or should we prioritize one for the initial version?
2. **Target Age Group:** What's the priority (Toddlers/Preschoolers/Elementary/Preteens)? Should we focus on one first?
3. **Image Generation:** Should we keep DALL-E image generation (requires API key + costs)? Or remove it for now?
4. **User Interface:** How much of the existing web UI should we reuse? Or should we rebuild with PyQt?
5. **Educational Topics:** Are there specific subjects to prioritize (science, math, language, stories)?

### Team Coordination
1. How should Yash and I divide the work?
   - Suggested split: I handle LLM/backend, Yash handles PyQt/interface?
2. Do we have shared resources (GitHub repo, cloud storage)?
3. Meeting schedule with you (weekly? bi-weekly?)
4. Communication channel (email, Slack, Discord)?

### Timeline & Milestones
1. What's the deadline for this project?
2. Key milestones you expect us to hit?
3. When do you want a working demo?
4. Will this become a paper/publication?

---

## Recommended Next Steps (Priority Order)

### Immediate (This Week)
1. **Get answers to questions above** at Thursday's meeting
2. **Set up development environment:**
   - Install Python dependencies
   - Download Phi-Ed model from Hugging Face
   - Test basic model inference locally
3. **Access Reachy Mini:**
   - If hardware available: Schedule time to experiment
   - If not: Set up simulation environment
4. **Meet with Yash:**
   - Divide responsibilities
   - Set up shared workspace (Git repository)
   - Create communication workflow

### Short-term (Weeks 2-3)
1. **Test Phi-Ed Model:**
   - Run inference with existing prompts
   - Compare quality to GPT-3.5 outputs
   - Measure response time and memory usage
2. **Prototype Reachy Mini Integration:**
   - Test basic movements (head, speech)
   - Study example apps on Hugging Face
   - Create simple "hello world" interaction
3. **Design System Architecture:**
   - Plan how components fit together
   - Define interfaces between modules
   - Create implementation timeline

### Medium-term (Weeks 4-6)
1. **Implement Core Functionality:**
   - Replace OpenAI calls with Phi-Ed inference
   - Rewrite robot control for Reachy Mini
   - Adapt WebSocket server if needed
2. **Test Integration:**
   - End-to-end story generation
   - Question/answer flow
   - Personality expression
3. **Build Interface (Yash):**
   - PyQt designer work
   - Teacher controls
   - Session management

### Long-term (Weeks 7+)
1. **Refine and Optimize:**
   - Response quality improvements
   - Performance optimization
   - Error handling
2. **User Testing:**
   - Internal testing with team
   - Pilot with small group
   - Iterate based on feedback
3. **Documentation & Deployment:**
   - User manual for teachers
   - Setup guide
   - Deployment to classroom

---

## Risk Assessment

### Technical Risks
- **GPU Memory:** Phi-Ed might not fit in available GPU memory
  - Mitigation: Use quantization, cloud GPUs if needed
- **Robot Availability:** If no hardware, delays testing
  - Mitigation: Use simulation, prioritize software development
- **Model Quality:** Phi-Ed might not match GPT-3.5 quality
  - Mitigation: Additional fine-tuning, hybrid approach

### Project Risks
- **Scope Creep:** Too many features, not enough time
  - Mitigation: Clear priorities, MVP approach
- **Team Coordination:** Splitting work with Yash
  - Mitigation: Regular check-ins, clear interfaces
- **Timeline:** Underestimating complexity
  - Mitigation: Weekly progress reviews, adjust scope

---

## Personal Learning Goals

1. Master QLoRA fine-tuning technique
2. Gain experience with embodied AI (robot interactions)
3. Understand educational HRI research methodology
4. Learn to deploy LLMs in real-world applications
5. Contribute to open-source robot ecosystem

---

## Resources & Links

**Project Resources:**
- GitHub: https://github.com/dtozadore/MI2US_Year2s
- Phi-Ed Model: https://huggingface.co/Mortadha/Phi-Ed-25072024
- Reachy Mini Docs: https://huggingface.co/docs/reachy_mini/
- PyQt Designer: https://www.pythonguis.com/installation/install-qt-designer-standalone/

**My Documentation:**
- All guides saved in project folder
- Git repository: [Your GitHub URL - share at meeting]
- Progress tracked in README TODO list

---

## Conclusion

I'm excited about this project and the opportunity to work with educational robotics and LLMs. I've completed the initial research and setup, and I'm ready to start implementation after our Thursday meeting.

The existing codebase is well-structured, and I believe we can successfully migrate from OpenAI GPT-3.5 + Alpha Mini to Phi-Ed + Reachy Mini while improving the system's autonomy and reducing operational costs.

I look forward to discussing the questions above and getting started with hands-on development!

---

**Prepared by:** Muhammad Maaz  
**Date:** February 10, 2026  
**Contact:** [Your Email]

---

## Appendix: File Inventory

### Project Files (Local)
```
Outreach Project/
├── README.md (307 lines) - Main documentation
├── CODEBASE_ANALYSIS.md - Code deep dive
├── REACHY_MINI_GUIDE.md - Robot documentation
├── QLORA_GUIDE.md - Fine-tuning guide
├── PROGRESS_REPORT.md - This document
├── Resources.md - Quick links
├── .gitignore - Security (excludes PDFs)
├── Automated_Robot_Personality_Traits_in_Educational_HRI.pdf (NOT on GitHub)
├── ROMAN_25__RobboBuddy_in_the_Classroom.pdf (NOT on GitHub)
└── MI2US_Year2s/ - Previous project code
    ├── Interface/ - Web app + Python backend
    ├── Experiment/ - Study data
    └── PhinetunEd/ - Fine-tuning resources
```

### GitHub Repository
- Clean (no unpublished papers)
- Includes: Code, README, documentation guides
- Ready to share with collaborators

---

*End of Report*
