# Codebase Overview

This document provides a high-level overview of the Whisplay AI Chatbot codebase, explaining the role of key files and directories.

## Project Structure

### Root Directory
- **`build.sh`**: Compiles the TypeScript code.
- **`install_dependencies.sh`**: Installs system-level dependencies (apt-get) and project dependencies (npm, pip).
- **`run_chatbot.sh`**: The main script to start the chatbot application.
- **`startup.sh`**: Configures the chatbot to run automatically on system boot.
- **`upgrade-env.sh`**: Helper script to update the `.env` file with new configuration options.
- **`package.json`**: Node.js project configuration, scripts, and dependencies.
- **`tsconfig.json`**: TypeScript compiler configuration.

### `src/` (TypeScript Source Code)
This directory contains the main application logic running on Node.js.

- **`index.ts`**: The entry point of the application. It initializes the `ChatFlow` and battery monitoring.
- **`test/`**: Test scripts.
  - **`ollama-text-test.ts`**: CLI tool to test Ollama response speed with text input.
- **`core/`**: Contains the core conversational logic.
  - **`ChatFlow.ts`**: The main state machine handling the conversation loop (listening -> processing -> speaking). It orchestrates audio recording, ASR, LLM interaction, and TTS.
  - **`StreamResponser.ts`**: Handles functionality for streaming responses from the LLM to the TTS engine.
  - **`Knowledge.ts`**: Manages context and system prompts for the LLM.
- **`cloud-api/`**: Interfaces and implementations for various AI services.
  - **`server.ts`**: The factory module that selects the active ASR, LLM, TTS, and Image Generation services based on environment variables.
  - **`interface.ts`**: Defines the interfaces that all service providers must implement.
  - **`gemini/`, `openai/`, `grok/`, `tencent/`, `volcengine/`**: Implementations for specific cloud providers.
  - **`local/`**: Implementations for local AI services (Ollama, local Whisper, Piper TTS, Vosk).
- **`config/`**: Configuration files.
  - **`llm-config.ts`**: Configuration for different LLM models.
  - **`llm-tools.ts`**: Definition of tools (functions) that the LLM can call.
  - **`custom-tools/`**: Directory for user-defined tools.
- **`device/`**: Modules for hardware interaction.
  - **`audio.ts`**: Handles audio recording and playback.
  - **`display.ts`**: Communicates with the Python display service to update the screen.
  - **`battery.ts`**: Monitors battery level.
- **`utils/`**: Utility functions for file handling, image processing, strings, etc.

### `python/` (Python Hardware Interface)
This directory contains Python scripts that interface directly with the hardware (Display, GPIO, Camera).

- **`chatbot-ui.py`**: The Python display service. It renders the UI on the LCD screen using Pillow and communicates with the Node.js application via a socket.
- **`whisplay.py`**: A low-level driver for the Whisplay hardware (LCD, LEDs, Buttons) using `RPi.GPIO` and `spidev`.
- **`camera.py`**: Handles camera capture functionality.
- **`led.py`**: Controls the RGB LEDs.
- **`requirements.txt`**: Python dependencies.

### `docker/` (Self-Hosted Services)
Configurations for running local AI services using Docker.

- **`faster-whisper-http/`**: Docker setup for running a local Faster Whisper ASR server.
- **`piper-http/`**: Docker setup for running a local Piper TTS server.

### `scripts/`
Helper scripts for setup and maintenance.

- **`install_ollama.sh`**: Script to install Ollama for local LLMs.
