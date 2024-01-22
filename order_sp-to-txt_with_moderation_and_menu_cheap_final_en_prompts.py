#Real time translation from Greek to English https://www.geeksforgeeks.org/create-a-real-time-voice-translator-using-python/
#Information about speech recognition - Not used for now https://www.askpython.com/python-modules/speech-recognition
#Information about using chatGPT https://platform.openai.com/docs/guides/chat/chat-vs-completions
#Other translation options (not tested) - https://github.com/BaseMax/UnlimitedAutoTranslate, https://www.deepl.com/en/pro-api
#Other Greek text-to-speech option - https://github.com/kostasx/greektext2speech - pyttsx3 does not work for Greek
#For running TTS and continue execution below it at the same time see https://superfastpython.com/run-function-in-new-thread/
#Using LLM to modify greek translation output of LLM chatbot does not work well.
#Another approach (not tested) is prompt with example of how the response of the LLM should look like
#Another option is to use Greeklish as input to an LLM and then translate the output back to Greek. OpenAI does not work well with Greeklish (doesn't make sense)
#and can also be slow sometimes. - https://stackoverflow.com/questions/59552782/how-to-convert-characters-from-greek-to-english-python

#Python version is 3.10 - Implementation ~3-4k lines of code

# pip install SpeechRecognition==3.10.0
# pip install googletrans==4.0.0-rc1
# pip install easygoogletranslate==0.0.4 #Needed as an alternative from gTTS for Greek text-to-speech
# pip install gTTS==2.3.2
# pip install gTTS-token==1.1.4
# pip install feedparser==6.0.10
# pip install playsound==1.2.2
# python -m pip install pyaudio==0.2.13
# pip install ipython==8.14.0
# pip install langchain==0.0.195
# pip install termcolor
# pip install openai==0.27.8
# pip install tee==0.0.3

# Below libraries for embeddings (include menu):

# pip install nltk==3.8.1
# pip install tiktoken==0.4.0
# pip install qdrant-client==1.2.0 #VectorStore
# pip install cohere==4.11.2 #For embeddings

#If a docsearch different than Qdrant is used, install the following:
# pip install faiss-cpu==1.7.4
# pip install chromadb==0.3.26
# pip install deeplake==3.6.6
# pip install pinecone-client==2.2.2

# Importing necessary modules required - Needs 3.5secs to load them all
import time
start=time.time()

import warnings
warnings.filterwarnings("ignore")

import speech_recognition as sr #Can't recognize questions!
# from googletrans import Translator
from easygoogletranslate import EasyGoogleTranslate
from gtts import gTTS
from playsound import playsound
import openai
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory # Chat specific components

import os
import traceback
from termcolor import colored
# from IPython.display import display, Markdown, Latex
from threading import Thread

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file with OpenAI key

#Set API keys for OpenAI and Cohere embeddings
openai.api_key = openai_api_key=os.environ['OPENAI_API_KEY']
cohere_api_key = os.environ['COHERE_API_KEY']

#Acapela Dependencies for Greek text-to-speech
import sys
import urllib
import re
import feedparser
from subprocess import call
from urllib.request import urlopen

#Embeddings and menu related dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import VectorStoreRetrieverMemory
#Explanation of some common types of memory: https://github.com/pinecone-io/examples/blob/master/generation/langchain/handbook/03-langchain-conversational-memory.ipynb
#'ConversationSummaryBufferMemory' that summarizes chat also used but not helpful since small conversation and not saving any tokens. 
#Detailed list of memory options in https://api.python.langchain.com/en/latest/modules/memory.html#

# from langchain.chains import ConversationalRetrievalChain, ConversationChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
#Explanation of each QA option: https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a

from langchain.embeddings import CohereEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings

#Detailed list in https://api.python.langchain.com/en/latest/modules/embeddings.html
#For HuggingFace Embeddings https://medium.com/@ryanntk/choosing-the-right-embedding-model-a-guide-for-llm-applications-7a60180d28e3

from langchain.vectorstores import Qdrant
# from langchain.vectorstores import FAISS
# from langchain.vectorstores import Chroma
# from langchain.vectorstores import DeepLake

#More details on how to use Qdrant on https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/qdrant
#Detailed list in https://api.python.langchain.com/en/latest/modules/vectorstores.html. Pinecone (save in cloud) tried but gave errors
#Some other suggestions are Postgres with pgvector extension or supabase. None of them works (require extra parameters).
#Suggestions taken from https://www.reddit.com/r/LangChain/comments/12ia7nc/what_is_the_best_vectorstore_to_selfhost_your/

import tiktoken

# import sys
# # Redirect standard output to a file
# sys.stdout = open('output.txt', 'w', encoding='utf-8')

try:
	#Create the menu by combining all txt files in one list. It does not worth translating to English first since embeddings search will not work better with it.
	documents=[]
	for file in os.listdir('./number_one_menu/final_best_with_GPT/'):
		if 'final_' in file and 'final_website_preprocessed' not in file:
			with open('./number_one_menu/final_best_with_GPT/' + file, 'r', encoding="utf8") as f:
				documents.append(f.read())
	# print('Menu is', documents)

	#Make each element in the list being an item from the menu - Does not work well with embeddings, needs more context
	# documents_final=[]
	# longest_text=0
	# for text in documents:
	#     text = text.split('\n')
	#     for item in text:
	#         documents_final.append(item) #Get each item of menu as an element of list
	#         if len(item)>longest_text:
	#             longest_text=len(item) #Will be ~175 characters

	# documents=documents_final #Replace our documents with the list of menu items

	# Make some changes in the menu
	for ind, doc in enumerate(documents):
		if 'Κλασσική' or 'κλασσική' in doc:
			documents[ind] = doc.replace("Κλασσική", "πίτα" )
			documents[ind] = doc.replace("κλασσική", "πίτα" )

	# Count number of tokens before sending to OpenAI (text-embedding-ada-002-v2 (price 0.0004$/1K tokens)) and calculate price of embedding.
	# - Source: https://www.datasciencebyexample.com/2023/04/10/count-tokens-in-gpt-models-accurately/

	# An alternative way is to use the 'get_num_tokens' function from langchain OpenAI
	# - Source: https://api.python.langchain.com/en/latest/modules/llms.html#langchain.llms.OpenAI.get_num_tokens

	# # Detailed list of pricing here: https://openai.com/pricing#language-models
	# all_docs="".join(documents) #Combine all texts to a string (needed below)

	# encoding = tiktoken.encoding_for_model("text-embedding-ada-002") #Returns the encoding

	# # calculuate the number of tokens 
	# num_token = len(encoding.encode(all_docs))

	# print("Number of tokens: ", num_token)
	# print("Total price: ", round(num_token*0.0004/1000,2),"$")

	# llm=OpenAI(openai_api_key=openai_api_key,temperature=0)
	# num_token=llm.get_num_tokens(all_docs)
	# print("Number of tokens (OpenAI function): ", num_token)

	# print("In comparison, total length of texts: ", len(all_docs))

	#Create embeddings of the menu to be processed faster and cheaper
	# Get your text splitter ready
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) #chunk_size refers to characters, not tokens! Initially set to 1000,200
	# Split your documents into texts
	texts = text_splitter.create_documents(documents)

	#Turn your texts into embeddings
	# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,model='text-embedding-ada-002') #Alternative option (cheap but not free)
	embeddings = CohereEmbeddings(model = "multilingual-22-12", cohere_api_key=cohere_api_key)
	#Taken from #https://txt.cohere.com/search-cohere-langchain/
	#Cohere Free tier limits: https://docs.cohere.com/docs/going-live
	#Information about OpenAI embeddings: https://platform.openai.com/docs/guides/embeddings

	#Get your docsearch ready
	docsearch = Qdrant.from_documents(texts, embeddings,location=":memory:",  collection_name="my_documents", distance_func="Dot") # Local mode with in-memory storage only
	# docsearch = FAISS.from_documents(texts, embeddings) #Performance similar to 'Qdrant'
	# docsearch = Chroma.from_documents(texts, embeddings) #Seems worse than FAISS/Qdrant
	# docsearch = DeepLake.from_documents(texts, embeddings) #Seems worse than FAISS/Qdrant
	# For very long documents embeddings may not work well: https://github.com/hwchase17/langchain/issues/2442

	#If we want to save them locally and load them for retrieval (Not used since quite fast for such small documents)
	# docsearch.save_local("faiss_midjourney_docs")
	# retriever=FAISS.load_local("faiss_midjourney_docs", OpenAIEmbeddings()).as_retriever(search_type="similarity", search_kwargs={"k":1})


	url = "http://vaassl3.acapela-group.com/Services/Synthesizer" #Acapela url for Greek text-to-speech

	class bcolors:
		HEADER = "\033[95m"
		OKBLUE = "\033[94m"
		OKGREEN = "\033[92m"
		WARNING = "\033[93m"
		FAIL = "\033[91m"
		ENDC = "\033[0m"

	def acapela( text = "δοκιμή", filename = 'voice.mp3' ):

		values = {
			'cl_env' : 'FLASH_AS_3.0',
			'req_snd_kbps' : 'EXT_CBR_128',
			'cl_vers' : '1-30',
			'req_snd_type' : '',
			'cl_login' : 'demo_web',
			'cl_app' : '',
			'req_asw_type' : 'INFO',
			'req_voice' : 'dimitris22k',
			'cl_pwd' : 'demo_web',
			'prot_vers' : '2',
			'req_text' : text
			}

		data = urllib.parse.urlencode(values).encode('utf-8')
		
		headers = { 
			"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0",
			"Content-Type" : "application/x-www-form-urlencoded",
			"Host" : "vaassl3.acapela-group.com" 
			}

		req = urllib.request.Request(url, data, headers)

		try:
			response = urllib.request.urlopen(req)
			page = response.read()
			match = re.search(r'http://(.*)mp3', page.decode('utf-8'))
			if match:
				urllib.request.urlretrieve(match.group(), filename)
		except urllib.error.URLError as e:
			print("Error", e)
		except urllib.error.HTTPError as e:
			print("HTTPError", e)


	#Inform the user that he/she cannot speak now and that information is being processed
	def wait_to_be_processed():
		wait_text='Παρακαλώ περιμένετε...'
		acapela(wait_text, filename='voice_wait.mp3')
		playsound('voice_wait.mp3') #Play the translated speech
		os.remove('voice_wait.mp3') #Remove the mp3 file


	# Capture Voice - takes command through microphone
	def takecommand():
		"""
		This function initializes a speech recognition object and listens for the user's input through a microphone.

		Returns:
		query (string): the user's spoken input, recognized by the speech recognition library. 
		If there are any issues with the audio or recognition, returns the string "None".
		"""

		r = sr.Recognizer() #Voice Recognition object
		with sr.Microphone() as source: #Microphone object
			print("Listening.....")
			r.pause_threshold = 1 #Seconds of non-speaking audio before a phrase is considered complete
			audio = r.listen(source) #Listen for audio

		try: #Try to recognize audio
			print("Recognizing.....")
			query = r.recognize_google(audio, language='el') #Language can be 'english' (or 'en'), 'greek' (or 'el')
			query=query.lower() #Make input to lowercase
			print(colored("Iteration_"+str(iteration) +"_Text:",'red'),query) #Print recognized text
			return query
		except Exception as e: #If error
			# print(traceback.format_exc()) #Print error information
			print("Say that again please.....")
			return "None"

		
	flag=0 #Denotes that we still taking the order
	flag_employee=0 #Denotes that the user wants to connect with an employee
	flag_address=0 #Denotes that the LLM is asking for an address - Don't translate address and avoid translating issues (eg. kolokotroni to pumpkin)
	iteration=0 #Denotes the number of iterations of listening to the user

	#Use above recognized text as input to an LLM to reformat it and take order/chat with the user
	#Menu should be included below to allow for better conversation

	#Price is ~0.25$ (12k tokens) for 10 inputs/outputs to the user +8 input/outputs for items in menu. Only the template costs ~0.01$ per call/user message!
	#To reduce cost of template, we tried to first encoded to a string (base64 - fewer tokens) and ask LLM to decode it. It didn't work since LLM outputs code.
	#If it outputs English text, this is made up and doesn't correspond to the text in the encoded string. 
	#We then tried to use langchain agents with code interpreter. This didn't work either and costs much more (2-2.5k tokens) than just using the original template.
	#At last, instead of using text DaVinci model we used gpt3.5-turbo which recently decreased in price and is much faster.

	#If any errors with 'end of order', we can try to use delimeter in the output (####).

	# Alternative template - Doesn't work that well here but can be used with DaVinci.
	# template = """
	# You are a helpful restaurant chatbot that takes an order from a user. You first greet the customer, then collect the order. 
	# Avoid making any unecessary comments and do not respond to the user with the items ordered so far.
	# You wait to collect the entire order and you keep track of items ordered so far, but not repeat them to the user. If the user asks questions try to answer it. 
	# If you don't know the answer or you are unconfident about something, say that you are connecting the user to an employee. Don't make things up. Don't give
	# answers if you don't have suffiecient information from the restaurant and instead connect the user with an employee.
	# You respond in a short, very conversational friendly style, without repeating any of the items ordered by the user so far. 
	# If the user asks, connect him/her with an employee.

	# When the user completes their order you ask them to say exactly one of the following two phrases 'pickup from the store' or 'delivery to address'. 
	# If any of these two phrases is not mentioned fully, ask them again to be more specific.
	# If it's for delivery, you ask for an address and for payment method (only card or cash). This question should be seperate from the one that provided the address.
	# If you get one of them, you only ask for the other. Never ask for the same information twice.
	# When these are provided, you note them down in the order and thank him/her for the order. 
	# Otherwise, you keep asking the user until that information is also provided. 
	# Do not continue the chat pretending to be the user/human and do not answer questions on his/her behalf.

	# If the conversion has finished, just thank the user for the order, without repeating the items ordered. Also, write in the final words of the output the phrase
	# 'end of order' without any other words after it. After that point, do not ask any further questions to the user and end the conversation.

	# If the order has not finished yet and an input from the user is expected, continue the chat with the user without outputting 'end of order'.
	# Do not continue the chat pretending to be the user/human and do not answer questions on his/her behalf.
	# In case that the user placed an order before and wants to modify it, by adding or removing anything, output the phrase 'connecting you with an employee'.


	template = """
	You are a helpful restaurant chatbot that takes an order from a user. You respond in a short, very conversational friendly style, without repeating
	any of the items ordered by the user so far. You response must have at least one question for the user, until the payment method is provided. 
	This is important so that the user knows what information needs to be provided to finalize the order.
	You must always respond in English, no matter in which language the input from the user is given or if there are pieces of text in other language. 
	Avoid asking for confirmation of any information provided by the user as well as asking the user if he changed their mind. 
	Avoid making any unecessary comments and repeating the same phrase everytime you respond to the user. Do not respond to the user with the items ordered so far.
	You wait to collect the entire order and you keep track of items ordered so far, but not repeat them to the user. If the user asks questions try to answer them. 
	Don't make things up. Don't give answers if you don't have sufficient information from the restaurant about items in menu and instead connect the user 
	with an employee. Don't continue the chat pretending to be the user/human and do not answer questions on his/her behalf.
	If the user asks, connect him with an employee. Avoid having in the output roles like 'human' or 'AI' or 'user' or any other role that clarifies who said what. 
	Just output your response. Give only one response and then wait for the user to continue the conversation. Do not make up the conversation. 
	Wait until you receive an input from the user before responding to it.  

	When the user completes his order or when he mentions that he doesn't want to add any more items in it,
	you ask him to say exactly one of the following two phrases 'pickup order from the store' or 'delivery of order to an address'. 
	If it's for delivery, you ask for an address. This will be provided in the form 'Address: <Greek name> <number>'.
	After you get an answer in that form, consider the address as provided. If you get a Greek response with a number, consider that as the address.
	After obtaining the address (if it's for delivery), you ask for payment method (only card or cash). You don't ask for card information.
	The answer should be either 'card' or 'cash'.
	Never ask for the same information twice. When these are provided, you note them down in the order and thank the user for the order. 
	Avoid providing the total price of the order or any other information after thanking him.
	The conversion ends at that point, and you should thank the user for the order, without repeating any of the items ordered so far and without 
	asking any further questions. At the end of the thank sentence, you should write in the final words of the output the phrase
	'end of order' without any other words after it. Make sure that this phrase is written once so that the chat will stop.

	If the order has not finished yet and an input from the user is expected, continue the chat with the user without outputting 'end of order'.
	Do not continue the chat pretending to be the user/human and do not answer questions on his behalf.
	"
	{chat_history}

	### Customer Input: {human_input}
	### AI Response:
	"""

	#Try the following as quick test: '3 souvlakia kotopoulo me sos', '1 patates me sos', '1 coca cola', '1 sprite', '2 (souvlakia) xoirina', '1 pita gyro apo ola',
	# '1 pita souvlaki kotopoulo patata ntomata sos (won't work)', '1 astakomakaronada', '3 podia kamilopardalis'

	# Initialize the LLMChain object
	prompt = PromptTemplate( 
		input_variables=["chat_history", "human_input"], 
		template=template)

	memory = ConversationBufferMemory(memory_key="chat_history") 
		
	llm_chain = LLMChain(
		llm=OpenAI(openai_api_key=openai_api_key,temperature=0,request_timeout=9,max_retries=1,model_name='gpt-3.5-turbo'), #'-16k' at the end if longer context needed
		prompt=prompt, 
		verbose=False, 
		memory=memory)
	#The above take no time to initialize

	while flag==0 and flag_employee==0: #While we are still taking the order and the user don't want to connect with an employee

		if iteration==0: #If first iteration

			# speak = gTTS(text='Καλέσατε το number one. Αν επιθυμείτε να μιλήσετε με έναν υπάλληλο παρακαλώ πείτε "σύνδεση με υπάλληλο". \
			#    Παρακαλώ δώστε την παραγγελία σας και πείτε αν επιθυμείτε παραλαβή από το κατάστημα ή delivery.',
			#     lang='el', slow=False) #Initializes instantly

			intro_text='Καλέσατε το number one. Αν επιθυμείτε οποιαδήποτε στιγμή να μιλήσετε με έναν υπάλληλο, παρακαλώ πείτε "σύνδεση με υπάλληλο". \
				Παρακαλώ δώστε την παραγγελία σας.' 

			acapela(intro_text)
			print(intro_text)

			playsound('voice.mp3') #Play the translated speech
			os.remove('voice.mp3') #Remove the mp3 file

			# speak.save("init_talking.mp3") # Using save() method to save the translated speech in an mp3 file. Takes ~1.5 secs to save
			# playsound('init_talking.mp3') #Play the translated speech
			# os.remove('init_talking.mp3') #Remove the mp3 file

		# Input from user
		query = takecommand() #Take input from user
		no_talk=0 #If no input for 10 seconds connect to employee
		while (query == "None"): #If no input
			query = takecommand() #Take input from user
			no_talk+=1
			if no_talk==5:
				flag_employee=1
				break

		#If we have user input, say to the user to wait until we processing what he/she said - Execute function above and continue below as we say that to the user.
		if query!="None": #If we have user input
			# create a thread
			thread = Thread(target=wait_to_be_processed)
			# run the thread
			thread.start()

		if 'υπάλληλο' in query.lower(): #If the user asked to connect to an employee (in Greek)
			flag_employee=1
			print("Connect with employee")
			break #Stop program execution

		# # invoking Translator - It initializes instantly - Uses hacked version of Google API, sometimes different translation from website
		# translator = Translator(service_urls=['translate.google.com'])

		# #Trying to translate in English to use as input to an LLM until we get no errors. Takes ~1.3 secs to translate

		# text_to_translate = None
		# while text_to_translate is None:
		# 	try:
		# 		#Translating from src to dest
		# 		text_to_translate = translator.translate(query, dest='english',src='el')
		# 	except:
		# 		pass

		# print(colored("Iteration_"+str(iteration) +"_English Translation:",'green'),text_to_translate.text)

		try: #Do not translate address from Greek to English - 'try' since 'response' might not be defined yet
			if ('address' in response.lower() and ('provid' in response.lower() or 'delive' in response.lower()) and 
					('pay' not in response.lower()) and 'store' not in response.lower()):
				#or 'delive' in response.lower() but failed for ' Would you like to pickup your order from the store or have it delivered to an address?'
				#If the previous output from LLM asked the user for his/her address
				flag_address=1

			print("Response address check is",response.lower())

		except:
			pass	

		#Uses translation from google website but with limit of 5000 characters. Takes no time to initialize
		translator = EasyGoogleTranslate( 
			source_language='el',
			target_language='en',
			timeout=10)
		
		#Trying to translate in English to use as input to an LLM until we get no errors - Takes 0.6-1.3 secs
		text_to_translate = None
		while text_to_translate is None:
			try:
				if flag_address==1: #If the previous output from LLM asked the user for his/her address
					query = 'Address: '+query #Use the Greek address as input to an LLM
					text_to_translate = query
				else:				
					#Translating from src to dest
					text_to_translate = translator.translate(query)

					if 'tradition' or 'receipt' in text_to_translate.lower(): #Avoid translation error
						text_to_translate = text_to_translate.lower().replace('tradition','delivery')
						text_to_translate = text_to_translate.lower().replace('receipt','pickup')		

					if 'sauce' in text_to_translate.lower(): #Avoid translation wth 'pita patata ntomata sos'
						text_to_translate = text_to_translate.lower().replace('sauce','and sauce')

					if 'address'==text_to_translate.lower(): #If user just say 'address', algorithm thinks that exact address provided
						text_to_translate=text_to_translate.lower().replace('address','to an address')		

				if text_to_translate is not None:
					flag_address=0 #Reset flag_address

			except:
				pass

		#Print Translated text to be used as input to an LLM
		print(colored("Iteration_"+str(iteration) +"_English Translation:",'green'),text_to_translate)

		iteration+=1 #Increase the number of iterations by 1

		human_text=text_to_translate #Translated text - '.text' if googletrans used
		
		# Perform similarity search - Response time range 1 to 1.5 seconds
		# docs = docsearch.similarity_search(query) #If OpenAI text-embedding-ada-002-v2 is used (price 0.0004/1K tokens), embedding for ~27K tokens around ~0.01$.
		#set k=n to only get n most similar docs
		#If 'similarity_search_with_score' used then we get "Attribute error tuple has no attribute 'page_content'".
		#This can be fixed by setting docs = [item[0] for item in docs_and_scores] as suggested in https://github.com/hwchase17/langchain/issues/3790

		# Conversation with user by taking the whole or part (based on similarity) of menu as input - If the whole menu is used, it is too expensive, 
		# 0.06$ per message/call since used ~3K tokens with 0.02 per 1K (assumed that documents without an item per line used)
		# - Information on how to use 'load_qa_chain': https://python.langchain.com/docs/use_cases/question_answering.html
		# - Add memory to a chain with multiple inputs: https://python.langchain.com/docs/modules/memory/how_to/adding_memory_chain_multiple_inputs
		# - Templates, prompts, and memory should have specific keys based on which chain is used: https://github.com/hwchase17/langchain/issues/1800

		response=llm_chain.predict(human_input=human_text) #Predict response using LLM - Takes ~1-5.5 secs to run through GPT 3.5 turbo
		print(colored("LLM response:",'yellow')+str(response))

		#Delete any further dialogue output from LLM
		if 'human:' in response.lower():
			print("Human part kept:",response.lower().split('human:'))
			response=response.lower().split('human:')[1]
		if 'chatbot:' in response.lower():
			print("chatbot part kept:",response.lower().split('chatbot:'))
			response=response.lower().split('chatbot:')[1]
		if 'ai:' in response.lower():
			print("AI part kept:",response.lower().split('ai:'))
			response=response.lower().split('ai:')[1]

		if 'employee' in response.lower(): #If LLM decides that the user should be connected to an employee
			flag_employee=1
			print("After LLM output, connect with employee")
			break #Stop program execution

		# #Trying to translate response bask in greek to speak it to the user until we get no errors - Takes ~1.5 secs to translate back to Greek
		# response_greek = None
		# while response_greek is None:
		# 	try:
		# 		#Translating from src to dest
		# 		response_greek = translator.translate(response, dest='el',src='english')
		# 	except:
		# 		pass

		#Trying to translate response back in greek to speak it to the user until we get no errors
		translator = EasyGoogleTranslate(
				source_language='en',
				target_language='el',
				timeout=10
			)

		response_greek = None
		while response_greek is None: #While we get errors in translation keep trying to translate
			try:
				#Translating from src to dest
				response_greek = translator.translate(response)

				if 'ξωτικό' or 'ξωτικο' or 'ξωτικά' or 'ξωτικα' in response_greek.lower(): #Avoid translation error
					response_greek = response_greek.lower().replace('ξωτικό','sprite')
					response_greek = response_greek.lower().replace('ξωτικο','sprite')
					response_greek = response_greek.lower().replace('ξωτικά','sprite')
					response_greek = response_greek.lower().replace('ξωτικα','sprite')

				if 'χοιρινά' or 'χοιρινα' in response_greek.lower() and ('μπριζολ' not in response_greek.lower() and 'σουβλ' not in response_greek.lower()):
					response_greek = response_greek.lower().replace('χοιρινά','σουβλάκια χοιρινά')
					response_greek = response_greek.lower().replace('χοιρινα','σουβλάκια χοιρινά')

				if 'εξαιρετική!' or 'εξαιρετικός!' or 'εξαιρετικη!' or 'εξαιρετικος!' in response_greek.lower():
					response_greek = response_greek.lower().replace('εξαιρετική','εξαιρετικά')
					response_greek = response_greek.lower().replace('εξαιρετικη','εξαιρετικά')
					response_greek = response_greek.lower().replace('εξαιρετικός','εξαιρετικά')
					response_greek = response_greek.lower().replace('εξαιρετικος','εξαιρετικά')

				if 'τέλειος!' or 'τελειος!'  in response_greek.lower():
					response_greek = response_greek.lower().replace('τέλειος','τέλεια')
					response_greek = response_greek.lower().replace('τελειος','τέλεια')

			except:
				pass

		print(colored("Say to user:",'magenta')+str(response_greek)) #'.text' if googletrans used for translation

		# Using Google-Text-to-Speech gTTS() method to speak the translated text into the destination language
		# Also, we have given 3rd argument as False because by default it speaks very slowly
		# Greek voice cannot change - No other options available in https://gtts.readthedocs.io/en/latest/module.html#localized-accents
		acapela(response_greek)
		playsound('voice.mp3') #Play the translated speech
		os.remove('voice.mp3') #Remove the mp3 file

		# speak = gTTS(text=response_greek, lang='el', slow=False) #Initializes instantly - '.text' if googletrans used
		# # Using save() method to save the translated speech in an mp3 file
		# speak.save("talking.mp3") #Takes ~1.5 secs to save
		# playsound('talking.mp3') #Play the translated speech
		# os.remove('talking.mp3') #Remove the mp3 file

		if 'end of order' in response.lower(): #If the order is completed
			flag=1 #Set flag to 1

			summary="Summarize the order above and return it as bullet points. Note down any special remarks comments or preferences of the user in a separate list. \
				Also note down the payment method, and if it's for pickup from the store or delivery, along with the address (if provided). \
				You should also note if user used innapropriate words or said something that is not usually said when someone orders food. \
				If not, then do not mention that point. Start the output with the bullet points as if you didn't have to give an answer to the user. \
				Be careful not to confuse ingredients that should be contained within an item as separate items."

			response_final=llm_chain.predict(human_input=summary) #Predict response using LLM - Takes ~1-5.5 secs to run through GPT 3.5 turbo
			print(colored("LLM response - order summary:",'yellow')+str(response_final))
			print('\n')

			if 'coke' in response_final.lower(): #Avoid translation error
				response_final=response_final.replace('coke','coca-cola')

			# #Content moderation - check for hate speech, harassment, and inappropriate words
			# moderation_check='' #Empty string to be filled with human messages
			# for user_message in memory.load_memory_variables({})['chat_history'].split('\n'): #Loop over Human and AI messages in memory
			# 	if 'Human:' in user_message and 'Summarize the order above and return it as bullet points.' not in user_message: #If human message and not summary
			# 		moderation_check=moderation_check+ user_message.split('Human: ')[1]+' \n ' #Add that message to a string to be used below for moderation check

			# #Check input to see if it flags the Moderation API or is a prompt injection
			# response = openai.Moderation.create(input=moderation_check) #Run moderation API
			# print(response)

			# moderation_output = response["results"][0] #Get moderation output

			# if moderation_output["flagged"]: #If input flagged by Moderation API
			#     # print("Input flagged by Moderation API")
			# 	categories_moderated = [] #Empty list to be filled with categories of inappropriate words
			# 	for key,val in response["results"][0]["categories"].items(): #Loop over categories
			# 		if val==1: #If category is inappropriate
			# 			categories_moderated.append(key) #Add category to list

			# 	print(translator.translate("In user's input there were the following issues"+ ', '.join(categories_moderated))) #Translate to Greek
			# 	print('\n')


			response_greek_final = None
			while response_greek_final is None: #While we get errors in translation keep trying to translate
				try:
					#Translating from src to dest
					response_greek_final = translator.translate(response_final) #Already set above to translate from English to Greek

					if 'ξωτικό' or 'ξωτικο' or 'ξωτικά' or 'ξωτικα' or 'κόκα' or 'κοκα' or 'κόλα' or 'κολα' in response_greek_final.lower(): #Avoid translation error
						response_greek_final = response_greek_final.lower().replace('ξωτικό','sprite')
						response_greek_final = response_greek_final.lower().replace('ξωτικο','sprite')
						response_greek_final = response_greek_final.lower().replace('ξωτικά','sprite')
						response_greek_final = response_greek_final.lower().replace('ξωτικα','sprite')
						response_greek_final = response_greek_final.lower().replace('κόκα','coca')
						response_greek_final = response_greek_final.lower().replace('κοκα','coca')
						response_greek_final = response_greek_final.lower().replace('κόλα','cola')
						response_greek_final = response_greek_final.lower().replace('κολα','cola')
					
					if 'στρογ' in response_greek_final.lower(): #Error with 'pita giro apo ola' translation
						response_greek_final = ' '.join([word for word in response_greek_final.split() if 'στρογ' not in word.lower()])


				except:
					pass

			print(response_greek_final) #Print the final order summary


			#Check to see if items available on menu
			#For 'πιτα' and 'τορτιγια' ingredients that will be filled with might not explicitly mentioned in menu.
			template_menu="""For the items ordered by the user provide a final price. If the item is not available in the menu, mention that explicitly in the output.
			If there is no item in the description also mention explicitly in the output that no item provided. Do not make up prices. If there is no such price
			in the menu, that means that the item does not exist. Items should be related to food. If they are unrelated just say that they are not available.
			{context}

			{chat_history}

			# Human: {human_input}
			# Chatbot:"""

			prompt_menu = PromptTemplate(input_variables=["chat_history", "human_input","context"], template=template_menu)  

			memory_menu = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

			#This method uses all text in menu
			llm_chain_menu = load_qa_chain(OpenAI(openai_api_key=openai_api_key,temperature=0, model_name='gpt-3.5-turbo'), 
					chain_type="stuff", memory=memory_menu, prompt=prompt_menu)
			# Another approach using RetrievalQA https://python.langchain.com/docs/modules/chains/popular/vector_db_qa
			# This method is the same as above. There would only be a difference if we haven't used the results of the similarity search above as input to the load_qa_chain.

			print("Now checking items in the following list:",response_greek_final.split('\n'))

			if 'coca-cola' in response_greek_final.lower():
				response_greek_final=response_greek_final.replace('coca-cola','coca cola')

			#For each item ordered by the user, perform similarity search and let the LLM decide the total price
			all_items=[]
			all_items_greek=[]

			if len(response_greek_final.split('\n'))<2: #In case all items returned in one line
				response_greek_final=response_greek_final.split('-')
				response_greek_final='\n'.join(response_greek_final)

			for item in response_greek_final.split('\n'):
				# if 'παράδοση' in item or 'πληρωμή' in item or 'παρατ' in item or 'παραγγ' in item or item=='\n' or item=='': #These are not items
				#above changed since last item could be in the same line as 'παρατ'
				if item=='\n' or item=='':
					pass
				else:
					print("Item is",item)
					
					if 'χοιρινά' or 'χοιρινα' in item.lower() and ('μπριζολ' not in item.lower() and 'σουβλ' not in item.lower()):
						item = item.lower().replace('χοιρινά','σουβλάκια χοιρινά')
						item = item.lower().replace('χοιρινα','σουβλάκια χοιρινά')

					print("List of similar items from menu:",docsearch.similarity_search(item,k=2))
					final_doc=docsearch.similarity_search(item,k=2)
					out=llm_chain_menu.run(input_documents=final_doc, human_input=item,  return_only_outputs=True, verbose=True) 
					#Run the chain - Response time ranges from 3 to 7 seconds

					print("LLM menu search result:",out)
					all_items.append(out) #Append all items to a list to get total price of order below

					response_greek_final_items = None
					while response_greek_final_items is None: #While we get errors in translation keep trying to translate
						try:
							#Translating from src to dest
							response_greek_final_items = translator.translate(out) #Already set above to translate from English to Greek

							if 'ξωτικό' or 'ξωτικο' or 'ξωτικά' or 'ξωτικα' in response_greek_final_items.lower(): #Avoid translation error
								response_greek_final_items = response_greek_final_items.lower().replace('ξωτικό','sprite')
								response_greek_final_items = response_greek_final_items.lower().replace('ξωτικο','sprite')
								response_greek_final_items = response_greek_final_items.lower().replace('ξωτικά','sprite')
								response_greek_final_items = response_greek_final_items.lower().replace('ξωτικα','sprite')

						except:
							pass

					print("Translation to Greek:",response_greek_final_items) #Print the final order summary
					all_items_greek.append(response_greek_final_items)

			all_items_greek_string='\n'.join(all_items_greek)
			print(colored("Greek order with menu check:",'blue')+str(all_items_greek_string))

			# # Get total price - Sometimes not work well
			# openai = OpenAI(openai_api_key=openai_api_key)
			# print("Total price of order is",openai('The following is an order placed by the user. What is the total price of all items in that order?'+all_items_string))

			end=time.time() #End time
			print("Time taken to take the order:",end-start)

			break #Break the loop
except:
	print(traceback.format_exc()) #Print error information
	print("Unexpected error - Connecting with employee")

# # Close the output file
# sys.stdout.close()

#########################################################################################
# # Using save() method to save the translated speech in an mp3 file
# speak.save("final_order_fast.mp3")

# #save output to txt file
# with open('output.txt', 'w',encoding="utf-8") as f:
# 	f.write(order)
##########################################################################################
#An Alternative way to do speech-to-txt it is by using whisper - For greek does not work well and takes much more time than above approach
# import whisper
# import torch
# model_path='./models_whisper/tiny.pt'

# #Whisper to transcribe audio of any language (or even translate to English)
# model = whisper.load_model('tiny',download_root=model_path+'ff') #small and above too much time for real-time inference
# result = model.transcribe("order_test.mp3", language='el', fp16=False) 
# #If language not English we should specify it if model!=large. We should also add 'task='translate''.
##########################################################################################
#If audio file is used as input
# SR_obj = sr.Recognizer() #Voice Recognition object
# info = sr.AudioFile("quick_test.wav") #Load audio file - only wav input

# with info as source:
#     SR_obj.adjust_for_ambient_noise(source) #adjust for ambient noise
#     audio_data = SR_obj.record(source) #Get audio data
    
# query=SR_obj.recognize_google(audio_data, language='el') #Recognize audio data
# print(colored("Text:",'red'),query) #Print recognized text
# query=query.lower() #Make input to lowercase
#########################################################################################
#We can also use OpenAI's content moderation to find inappropriate language used by user. 
#This takes time to run since it gets as input user messages (extra prompt). Can be implement at the end, when we get summary of order.
#If we do that, only give as input to LLM user messages. Otherwise it might be expensive. Below is the implementation of it:

#Check input to see if it flags the Moderation API or is a prompt injection
# response = openai.Moderation.create(input=user_input)
# print(response)

# moderation_output = response["results"][0]

# if moderation_output["flagged"]:
#     print("Step 1: Input flagged by Moderation API.")
#########################################################################################
#An alternative approach is to use OpenAI directly instead of langchain.
#In that case, some useful functions to create a chatbot, taken from the deeplearning.ai course are:

# def get_completion(messages, model="gpt-3.5-turbo"):
#     # messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0, # this is the degree of randomness of the model's output
#     )
#     return response.choices[0].message["content"]

# def collect_messages(context):
#     context.append({'role':'user', 'content':f"{context}"})
#     response = get_completion(context) 
#     print(response)
#     context.append({'role':'assistant', 'content':f"{response}"})
#     return context

# context = [ {'role':'system', 'content':"""
# Here you write down the prompt \
# It should be the big one, used above
# """} ]

#Then you chat with the user as follows:
# human_text="""Good evening I would like to order ...."""
# message1=collect_messages(human_text)
# print(message1)

#At the end, you can summarize the order with the following:
# system_text="Summarize the order of the user and return it in bullet points."
# system_text="Summarize the order ...."
# context.append({'role':'system', 'content':system_text})
# response = collect_messages(system_text) 
#########################################################################################
# TO DO:

# 1) response=..chatcompletion...(stream=True)
#     for chunk in response:
#         chunk_message = chunk['choices'][0]['delta']  # extract the message
# 2) Menu modifications: 
# OK - Coca cola in English and Greek
# OK - Avoid having extra material in items due to similarity search (eg. pita with giros sos might results also with onion etc. based on menu)
# OK - If a prompt has more than one item then need to first split it and then feed each one seperately to similarity check. Otherwise uses the text embedding where
#      one item was found to search for the second too. This might results in mading things up. Maybe better to feed each item in the final order list to similarity
#      check instead of doing that during the order.
# ?? - If item does not exist, inform the user. Also note if item removed/modified. 
# OK - Sometimes it asks things like what size of fries etc. due to menu absence
# ?? - 'pita souvlaki kotopoulo patata ntomata sos' doesn't work since it thinks that the sauce consists of potatoes and ntomatoes
# ?? - 'pita giro apo ola' will translate to 'στρογγυλη πιτα απο ολα' and won't be found in menu.
# 3) For price estimation, use agents? For example, if two same items are ordered (2 souvlakia xoirina), will it find the correct price? => Yes, so no need for agents
# 4) Add a general try except where we connect to an employee in case of any error (eg. OpenAI call failed, no sound available to speak etc.) => Done
# 5) Perform similarity search if user asks 'do you have souvlakia today?' - Can't be done since the speech recognition module doesn't recognize questions.

# 6) Set a maximum duration of conversation (to ensure we are not getting charged for fun) and then connect to employee, informing them of the duration. => Not done
# 7) Create prompts with examples eg. user ordered 2 xoirina, 3 kotopoulo then removed 2 kototpoulo so final order should be... => Not done
#########################################################################################
# Conclusions

# - Best approach is the load_qa_chain but relatively expensive. With new gpt3.5-turbo version just used LLMchain. 
# - Next best option is just similarity search and feed the result of the chat to the main conversation with user (where the actual order takes place with 
#   speech-to-text translation). In similarity search use k=1 since it's perform the best and is the cheapest option.
# - A very useful post with explanations on what embeddings are, what are vector stores, how to use them within langchain, along with advantages of langchain 
#   (makes it easier and implementation in fewer lines of code) can be found in: https://medium.com/sopmac-ai/vector-databases-as-memory-for-your-ai-agents-986288530443