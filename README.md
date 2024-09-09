# Towards Optimal RAG Project


This repository hosts the project "Towards Optimal RAG: Benchmarking Open-Source Extractive and Generative Q&A Models." The aim of this research is to evaluate the performance of open-source extractive and generative models in Retrieval-Augmented Generation (RAG) systems for Question Answering (Q\&A) tasks.

## Setup and Execution

### Option 1: Using Python Environment

1. **Create a Virtual Environment**
   ```bash
    conda create --name rag_env python=3.12
    conda activate rag_env
    ```

2. **Install Dependencies**
   ```bash
    pip install -r requirements.txt
    ```

1. **Run the Notebooks**
    Navigate to the runs/ directory and open the notebook you wish to run.


### Option 2: Using Docker

1. **Prerequisites:** Ensure you have Docker installed on your machine. If not, install it from [Docker's official website](https://www.docker.com/get-started).

2. **Make the Script Executable:** Before running the script, make sure it is executable. You can do this with the following command:
     ```bash
     chmod +x initialize_docker.sh
     ```

3. **Run the Initialization Script:** Execute the script to build the Docker image and run the container. This script will stop and remove any existing containers, build a new Docker image, and run the container in detached mode with appropriate settings.
     ```bash
     ./initialize_docker.sh
     ```

4. **Access Jupyter Notebook:** Open a web browser and go to `http://localhost:10002`. You will be directed to the Jupyter Notebook interface running inside the Docker container, where you can open and run the notebooks like `run_extractive_open_souce.ipynb` and `run_generative_open_source.ipynb`.

5. **Stopping the Container:** To stop the Docker container, use:
    ```bash
    docker stop optimal_rag
    ```