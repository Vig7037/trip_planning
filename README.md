# Trip Planning LLM Model for Uttarakhand

This project leverages the NVIDIA NIM platform and the Llama3-70b-instruct model to provide personalized trip planning for Uttarakhand. By using advanced AI techniques, the model generates detailed itineraries based on user inputs such as destination and duration.

## Features

- **AI-Powered Trip Planning**: Utilizes Llama3 to generate accurate and insightful trip plans.
- **NVIDIA NIM Integration**: Implements NVIDIA Embeddings and ChatNVIDIA for enhanced performance and accuracy.
- **Real-time Document Retrieval**: Efficiently retrieves and processes documents to provide precise recommendations and contextually relevant information.
- **Streamlit Web Application**: User-friendly interface for seamless interaction and trip planning.

## Setup

### Prerequisites

- Python 3.x
- Required Python libraries: `streamlit`, `langchain_nvidia_ai_endpoints`, `langchain_community`, `langchain_core`, `dotenv`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/uttarakhand-trip-planner.git
    cd uttarakhand-trip-planner
    ```

2. Install the required Python libraries:
    ```sh
    pip install streamlit langchain_nvidia_ai_endpoints langchain_community langchain_core python-dotenv
    ```

3. Set up the `.env` file with your NVIDIA API key:
    ```env
    NVIDIA_API_KEY=your_nvidia_api_key
    ```

### Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to use the trip planning assistant.

## Project Structure

- `app.py`: Main application file with Streamlit code.
- `.env`: Environment file for storing API keys.
- `requirements.txt`: List of required Python libraries.

## Contributors

- [Vighnesh Singhal](https://github.com/Vig7037)

Feel free to contribute and improve this project. If you encounter any issues or have suggestions for enhancements, please open an issue or submit a pull request.

Happy trip planning! 🏞️✈️🗺️
