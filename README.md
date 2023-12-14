# my_first_GPT_Model
Building my first GPT model based on bigram language model, LLM, NLP and semantics
This project implements a GPT (Generative Pre-trained Transformer) language model using PyTorch, trained on the OpenWebText dataset. The model generates text based on the patterns it learns from the input data.  

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project
   ```
2.Install dependencies:
  ```bash
  pip install torch numpy
  ```
  Install additional libraries required for the code.
  
### Usage
- Ensure you have the necessary dataset files in the specified directories (oz.txt, openwebtext).
- Run data_extract.py changing the directory of the dataset to your directory.
- Run MyFirstGPTModel.py for training the model
- Model is stored as model_01.pkl
- Run chatbot.py to test prompts and obtain results.

### Contributing
Feel free to contribute to this project. Please fork the repository, make changes, and submit a pull request.

Acknowledgments
The GPT model implementation is based on the paper "Attention Is All You Need."
PyTorch for the deep learning framework.
OpenWebText dataset for training data.
Reference to Elliot Arledge - 'project to build a LLM from Scratch in Python'.
