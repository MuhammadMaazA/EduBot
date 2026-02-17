# Reachy Mini Robot Guide

## Overview

Reachy Mini is an open-source, expressive robot designed for AI builders and hackers. It's the robot you'll be using for your educational project (replacing the Alpha Mini from the previous study).

**Official Documentation:** https://huggingface.co/docs/reachy_mini/index

---

## Hardware Versions

### 1. Reachy Mini (Wireless) - Full Autonomous Version
- **Computer:** Raspberry Pi 4 onboard
- **Power:** Battery powered
- **Connectivity:** WiFi
- **Sensors:** IMU (Inertial Measurement Unit)
- **Assembly Time:** 2-3 hours
- **Use Case:** Fully autonomous robot, no tethered connection

### 2. Reachy Mini Lite - Developer Version
- **Computer:** Runs on your PC (USB connection)
- **Power:** Wall outlet (AC powered)
- **Connectivity:** USB to your computer
- **Assembly Time:** 2-3 hours
- **Use Case:** Development and testing (cheaper, easier to debug)

### 3. Simulation - No Hardware Required
- **Platform:** MuJoCo physics simulator
- **Use Case:** Prototyping code before hardware arrives
- **Benefit:** Test your code without the robot

**Recommendation:** Start with simulation, then move to Lite for development, then Wireless for deployment.

---

## Key Features

1. **Expressive Design** - Can show emotions through movements
2. **Open Source** - Full hardware and software are open
3. **Camera** - Built-in vision capabilities
4. **Microphone & Speaker** - Can hear and speak
5. **Movable Head** - Pan/tilt for looking around
6. **Arms** - Can gesture and point
7. **LED Eyes** - Visual feedback

---

## Software Architecture

### Installation

**Quick Install (with uv - recommended):**
```bash
pip install uv
uv pip install reachy-mini
```

**Standard Install:**
```bash
pip install reachy-mini
```

### Basic Python API

#### 1. Connection
```python
from reachy_mini import ReachyMini

# Connect to robot
with ReachyMini() as mini:
    # Your code here
    pass
```

#### 2. Head Movement
```python
from reachy_mini.utils import create_head_pose

with ReachyMini() as mini:
    # Look up and tilt head
    mini.goto_target(
        head=create_head_pose(z=10, roll=15, degrees=True, mm=True),
        duration=1.0
    )
```

#### 3. Vision
```python
with ReachyMini() as mini:
    # Get camera frame
    frame = mini.get_camera_frame()
```

#### 4. Audio
```python
with ReachyMini() as mini:
    # Text-to-speech
    mini.speak("Hello, I am Reachy Mini!")
    
    # Listen
    audio_data = mini.listen()
```

---

## App Ecosystem

Reachy Mini has an **App Store** powered by Hugging Face Spaces. Apps can be installed with one click from the robot's dashboard.

### Available Apps

1. **Conversation App** - Talk naturally with the robot (LLM powered)
   - URL: https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app
   - Perfect for your educational storytelling project!

2. **Radio** - Listen to radio with Reachy
   - URL: https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio

3. **Hand Tracker** - Robot follows your hand movements
   - URL: https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2

**You can create and publish your own apps to Hugging Face!**

---

## Coordinate System

Reachy Mini uses 3D coordinates (x, y, z) and orientations (roll, pitch, yaw):

- **X-axis:** Left-right
- **Y-axis:** Forward-backward  
- **Z-axis:** Up-down

Head poses combine position and orientation for expressive movements.

---

## Safety Features

- **Built-in limits** prevent dangerous movements
- **Compliant design** - Safe for human interaction
- **Emergency stop** capabilities
- **Temperature monitoring** on motors

---

## Development Workflow

### Phase 1: Simulation (No Hardware)
```bash
# Install simulation
pip install reachy-mini[simulation]

# Run in MuJoCo
python your_script.py --simulation
```

### Phase 2: Reachy Mini Lite (Development)
```bash
# Connect via USB
# Run on your laptop
python your_script.py --lite
```

### Phase 3: Reachy Mini Wireless (Deployment)
```bash
# SSH into Raspberry Pi
ssh pi@reachy-mini.local

# Run autonomous code
python your_script.py
```

---

## Integration with Your Project

### What You'll Need to Implement

1. **Storytelling Mode**
   - Use `mini.speak()` for narration
   - Use head movements for expression
   - Sync with your story generation

2. **Question/Answer Interaction**
   - Use `mini.listen()` to hear student responses
   - Process with speech recognition
   - Use LLM (Phi-Ed) to generate responses
   - Use `mini.speak()` to ask questions

3. **Visual Engagement**
   - Display generated images on screen
   - Use head movements to "look" at images
   - Point with arms (if applicable)

4. **Personality Expression**
   - Map extraversion/introversion to movement styles
   - Extroverted: More animated, faster movements
   - Introverted: Subtle, slower movements

---

## Comparison: Alpha Mini vs Reachy Mini

| Feature | Alpha Mini (Old) | Reachy Mini (New) |
|---------|------------------|-------------------|
| Height | 24.5 cm | ~30 cm (estimated) |
| Platform | Proprietary SDK | Open source |
| Computer | External laptop | RPi 4 onboard OR PC |
| Community | Limited | Active (Discord, HF) |
| Docs | Basic | Comprehensive |
| Apps | None | App Store (HF Spaces) |
| Cost | Unknown | Check hf.co/reachy-mini |

**Key Advantage:** Reachy Mini is open-source with a growing community and app ecosystem.

---

## Resources

1. **Official Docs:** https://huggingface.co/docs/reachy_mini/
2. **Purchase:** https://www.hf.co/reachy-mini/
3. **Discord Community:** https://discord.gg/Y7FgMqHsub
4. **GitHub Repository:** https://github.com/pollen-robotics/reachy_mini
5. **Example Code:** https://github.com/pollen-robotics/reachy_mini/tree/main/examples
6. **App Publishing Guide:** https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps

---

## Quick Start Checklist

- [ ] Read official documentation
- [ ] Install reachy-mini package
- [ ] Try simulation mode (if no hardware yet)
- [ ] Test basic movements (head, speech)
- [ ] Study example apps on Hugging Face
- [ ] Plan your storytelling app architecture
- [ ] Join Discord community for support

---

## Questions to Ask Professor

1. Do we have access to Reachy Mini hardware yet?
2. Which version: Wireless or Lite?
3. Can we start with simulation while waiting?
4. Budget for purchasing if needed?
5. Lab space for testing with students?

---

*Document created: February 10, 2026*
*Based on official Reachy Mini documentation*
