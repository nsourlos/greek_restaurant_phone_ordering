# Automatic phone ordering in Greek


[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/uses-badges.svg)](https://forthebadge.com)

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)

## [Main script](./order_sp-to-txt_with_moderation_and_menu_cheap_final_gr_prompts.py)

> This script takes a phone order from a user on a restaurant. This call is performed in Greek. 

### Installation

To run this script, you need to install several Python libraries. You can install them using pip, a package installer for Python. Open your terminal and run the following commands:

```bash
pip install SpeechRecognition==3.10.0 
pip install googletrans==4.0.0-rc1
pip install easygoogletranslate==0.0.4
pip install gTTS==2.3.2
pip install gTTS-token==1.1.4
pip install feedparser==6.0.10
pip install playsound==1.2.2
python -m pip install pyaudio==0.2.13
pip install ipython==8.14.0
pip install langchain==0.0.195
pip install termcolor
pip install openai==0.27.8
python -m pip install python-dotenv
pip install nltk==3.8.1
```

### Usage 

```
python order_sp-to-txt_with_moderation_and_menu_cheap_final_gr_prompts
```

We should first set the OpenAI API key.

### Limitations
Using LLMs to modify Greek translation output of a LLM chatbot does not work well. It also mentions that OpenAI does not work well with Greeklish and can be slow sometimes. Moreover, the speech recognition component doesn't recognize user's questions.

### Future Work
The script includes several untested options for future development, including other translation options and other Greek text-to-speech options. It also suggests the possibility of using Greeklish as input to an LLM and then translating the output back to Greek.


### [order_evaluation_gr.ipynb](./order_evaluation_gr.ipynb)

> This Jupyter notebook is primarily used for evaluating the performance of a language model on a question-answering task.

### Installations

The notebook begins by importing necessary modules from the langchain package, which are used for various tasks such as text splitting, generating embeddings, storing and retrieving vectors, interacting with the OpenAI API, generating responses to prompts, retrieving answers to questions from a database, and evaluating the performance of a question-answering model.

The os and openai modules are imported for interacting with the operating system and the OpenAI API, respectively. The dotenv package is used for managing environment variables.

The notebook then loads environment variables from a .env file, which is expected to contain the API keys for OpenAI and Cohere. These keys are then set as the API keys for the respective services.

The overall function of this notebook is to set up the necessary environment and tools for evaluating a language model's performance in a question-answering task.

Below the imports used and their function are presented:

```bash
from langchain.text_splitter import RecursiveCharacterTextSplitter
```
- Split text into smaller chunks based on certain criteria.

```bash
from langchain.embeddings import CohereEmbeddings
```
- Generate embeddings of text using the Cohere API.

```bash
from langchain.vectorstores import Qdrant
```
- Store and retrieve vectors in a Qdrant database

```bash
from langchain.llms import OpenAI
```
- Interact with the OpenAI API.

```bash
from langchain.chat_models import ChatOpenAI
```
- Generate responses to prompts using the OpenAI API.

```bash
from langchain.chains import RetrievalQA
```
- Retrieve answers to questions from a database of precomputed answers.

```bash
from langchain.evaluation.qa import QAEvalChain
```
- Evaluate the performance of a question-answering model.

```bash
from dotenv import load_dotenv, find_dotenv
```
- Load environment variables from a .env file.

```bash
_ = load_dotenv(find_dotenv())
```
- Load the environment variables from a .env file in the current directory or any of its parents.





### [openai_embeddings_final_gr.ipynb](./openai_embeddings_final_gr.ipynb)

### Overview

The code is designed to create a chatbot that takes food orders based on a predefined menu. The chatbot utilizes various libraries and modules from the LangChain framework to implement conversational retrieval and question-answering capabilities. The chatbot processes user input, searches for relevant items in a menu, and provides appropriate responses.

The main ideas from this file were used in the main file to utilize embeddings (commented out for now). 

### Libraries and Modules

The code relies on several external libraries and modules. Make sure to install them using the provided pip commands before running the code.

```langchain```: A conversational AI framework. The modules used include text splitters, memories, conversational chains, embeddings, and vector stores.

```nltk```: Natural Language Toolkit for natural language processing tasks.

```tiktoken```: A Python library for counting tokens in a text string.

```openai```: OpenAI's Python library for accessing the GPT models.

```qdrant-client```: Client for Qdrant, a vector search engine.

```cohere```: Cohere's library for embeddings.

```dotenv```: A library to read variables from a local .env file.

### Notes and References
The code includes comments explaining various sections, functionalities, and choices made during implementation.
External documentation links are provided for further understanding of LangChain functionalities.
Suggestions for alternative vector stores and their limitations are mentioned based on community discussions.
Make sure to follow the provided instructions, including installing the required libraries and setting up the OpenAI API key, before running the code.

### [web_parsing_jupyter_with_GPT.ipynb](./web_parsing_jupyter_with_GPT.ipynb)

### Overview

The given code is designed to scrape information from multiple websites (specified by URLs), particularly extracting menu items along with their prices. It utilizes web scraping techniques, BeautifulSoup library for parsing HTML, and OpenAI's GPT-3.5-turbo model for refining the extracted information.

### Dependencies
```requests```: For making HTTP requests to fetch website content.

```BeautifulSoup``` from bs4: For parsing HTML and extracting relevant information.

```markdownify```: To convert HTML text to markdown format for simplicity.

```openai```: For interacting with the OpenAI GPT-3.5-turbo model.

### Steps

1. ```Fetching URLs:```
The code starts by fetching URLs from the provided website (https://www.numberone.gr/) using the BeautifulSoup library. It extracts all ```<a>``` tags from the HTML, ensuring that only URLs starting with 'https:' are considered relevant.
2. ```Web Scraping Function (pull_from_website):```
A function is defined to fetch and parse content from a given URL. It returns the text content of the HTML page after converting it to markdown.
3. ```Extracting Menu Items:```
The code iterates through each URL, extracts the text content using the ```pull_from_website``` function, and looks for relevant menu items. If found, it extracts them.
4. ```Formatting and Saving Data:```
The extracted menu information is formatted and saved in a text file. Additionally, the raw website data is accumulated in a string (```website_data```).
5. ```GPT-3.5-turbo Integration:```
The formatted menu information is sent to the GPT-3.5-turbo model using OpenAI's API. The system message instructs the model to extract each item from the menu along with its price in a specific format.
The model's response is saved in a text file, and the formatted text is appended to the ```final_data``` string.
6. ```Error Handling:```
The code includes error handling to manage cases where certain phrases or characters are not found during processing.
7. ```Output Files:```
The final accumulated website data is saved in ```final_website_preprocessed.txt```.
The GPT-3.5-turbo model's output is saved in ```final_GPT_output.txt```.
Individual menu information for each website is saved in separate text files.

### Notes
The code assumes a specific structure in the HTML of the websites being scraped.
The use of GPT-3.5-turbo is based on OpenAI's API, and the API key must be set before running the code.
The extracted information is processed to retain only the Greek descriptions, convert to lowercase, and format according to the specified guidelines.

We should ensure that the provided URLs lead to web pages with menu information in a consistent format.
Regularly check and update the code if the structure of the web pages being scraped changes.
Monitor API usage and adhere to OpenAI's guidelines and terms of service.


## Contributing
Contributions are welcome. Please open an issue first to discuss what you would like to change.

## License
Please check the license terms of the individual libraries used in this script. The code is freely available for non-commercial applications. For commercial use please feel free to ask for pricing.