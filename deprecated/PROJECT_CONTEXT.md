### 📋 O.A.S.I.S.

**[Project Overview]**

* **Project Name:** O.A.S.I.S. (Offline AI Survival & First-aid Kit)
* **Goal:** To build a fully offline, voice-interactive AI assistant for emergency first-aid situations on a Raspberry Pi 5.
* **Base Codebase:** Forked from `whisplay-ai-chatbot` (TypeScript/Node.js for core logic + Python for hardware control).

**[Hardware Environment]**

* **Device:** Raspberry Pi 5 (8GB RAM).
* **Power:** PiSugar (UPS/Battery module) for portability.
* **Display:** Waveshare 3.5-inch HDMI LCD (or similar generic HDMI/SPI display) - *Replacing the original Whisplay e-paper/LCD.*
* **Audio:** USB Microphone & Speaker (or 3.5mm/HDMI audio) - *Replacing the original Whisplay audio HAT.*
* **Network:** Must operate 100% offline (no internet connection required during runtime).

**[Software Stack & Architecture]**

* **OS:** Raspberry Pi OS (Bookworm, 64-bit).
* **Core Logic (Node.js):**
* `src/core/ChatFlow.ts`: Main conversation loop (ASR -> LLM -> TTS).
* `src/cloud-api/`: Interfaces for AI services.


* **Hardware Interface (Python):**
* Communicates with Node.js via Socket.IO/Unix Socket.
* Controls Display (UI rendering) and GPIO (LEDs/Buttons).


* **AI Engine (Local):**
* **LLM:** Ollama (running locally with Llama3 or BioMistral quantized models).
* **STT (ASR):** Faster-Whisper (local).
* **TTS:** Piper (local).
* **RAG:** LangChain + Vector DB (e.g., LanceDB or Faiss) for retrieving medical manuals.

* **Note**
* The existing codebase uses dedicated local HTTP servers for STT (Faster-Whisper) and TTS (Piper) to minimize latency.
* Instead of loading models directly within the main logic loop, they run as separate services (likely using Python/FastAPI or similar) and communicate via HTTP requests.

**[Key Modification Goals - What I need you to help with]**

1. **Hardware Decoupling:**
* The original code is tightly coupled with `Whisplay` hardware drivers.
* I need to refactor `python/chatbot-ui.py` and `python/whisplay.py` to support a generic Waveshare display and standard Linux ALSA audio instead of the specific HAT drivers.


2. **RAG Implementation:**
* Modify `src/core/Knowledge.ts` to implement a RAG (Retrieval-Augmented Generation) pipeline.
* The system should search a local medical knowledge base before sending a prompt to Ollama.


3. **Dockerization:**
* Replace the current shell script installation (`install_dependencies.sh`) with a `Dockerfile` and `docker-compose.yml`.
* The container needs to manage Node.js, Python, and the connection to the Ollama service.


4. **Performance Optimization:**
* Optimize latency for the Pi 5.
* Ensure VAD (Voice Activity Detection) is tuned for quick turn-taking.

**[Selected Technology Stack & Decisions]**

* **UI Framework:** Python + Pygame (Direct Framebuffer access for lightweight, stable rendering).
* **Interaction Mode:** Hybrid (Primary: Push-to-Talk for reliability/battery; Secondary: Wake Word detection).
* **RAG:** LangChain + LanceDB (Local file-based vector DB, no separate service required).
* **Containerization:** Multi-container Docker Compose (Node.js App, Python Hardware Service, Ollama AI Service all separated).
