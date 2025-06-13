# LLM Trading Simulation

This repository contains the source code for a trading simulation environment powered by Large Language Models (LLMs).

## Description

This project simulates a financial market where agents, powered by LLMs, make trading decisions.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd LLMTradingSimulation_public
    ```

2.  **Create a virtual environment:**
    It is highly recommended to use a virtual environment. For example, with conda:
    ```bash
    conda create -n llm_trading python=3.11
    conda activate llm_trading
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    This project requires an `OPENAI_API_KEY` to be stored in a `.env` file at the root of the project. This file is not tracked by Git for security reasons. 

    You will need to create this file yourself and add your key like this:
    ```
    OPENAI_API_KEY="sk-..."
    ```

## Usage

(Add instructions on how to run your simulation here) 