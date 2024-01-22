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

# Below libraries for embeddings (include menu):
# python -m pip install python-dotenv
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

#Save terminal output to txt file
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
	# 	text = text.split('\n')
	# 	for item in text:
	# 		documents_final.append(item) #Get each item of menu as an element of list
	# 		if len(item)>longest_text:
	# 			longest_text=len(item) #Will be ~175 characters

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

	# # calculate the number of tokens 
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
	flag_model=0 #To decice if context length is more than 4k and so, we should change to 16k model
	iteration=0 #Denotes the number of iterations of listening to the user

	#Use above recognized text as input to an LLM to reformat it and take order/chat with the user
	#Menu should be included below to allow for better conversation

	#Price is ~0.25$ (12k tokens) for 10 inputs/outputs to the user +8 input/outputs for items in menu. Only the template costs ~0.01$ per call/user message!
	#To reduce cost of template, we tried to first encoded to a string (base64 - fewer tokens) and ask LLM to decode it. It didn't work since LLM outputs code.
	#If it outputs English text, this is made up and doesn't correspond to the text in the encoded string. 
	#We then tried to use langchain agents with code interpreter. This didn't work either and costs much more (2-2.5k tokens) than just using the original template.
	#At last, instead of using text DaVinci model we used gpt3.5-turbo which recently decreased in price and is much faster.

	#If any errors with 'end of order', we can try to use delimeter in the output (####).

	template = """
    Είσαι ένα χρήσιμο chatbot εστιατορίου που δέχεται μια παραγγελία από έναν χρήστη. Απάντησε με σύντομο και φιλικό ύφος, χωρίς να επαναλαμβάνεις
    οποιοδήποτε από τα είδη που έχει παραγγείλει ο χρήστης μέχρι στιγμής. Η απάντησή σου πρέπει να έχει τουλάχιστον μία ερώτηση για τον χρήστη, μέχρι το σημείο όπου
    παρέχεται ο τρόπος πληρωμής. Αυτό είναι σημαντικό ώστε ο χρήστης να γνωρίζει ποιες πληροφορίες πρέπει να παράσχει για να οριστικοποιήσει την παραγγελία.
    Πρέπει πάντα να απαντάς στα Ελληνικά, ανεξάρτητα από τη γλώσσα στην οποία δίνεται η παραγγελία από τον χρήστη ή ακόμα και αν υπάρχουν μέρη της παραγγελίας 
    σε άλλη γλώσσα. Απόφυγε να ζητήσεις επιβεβαίωση οποιασδήποτε πληροφορίας που παρέχεται από τον χρήστη καθώς και να ρωτήσεις τον χρήστη εάν άλλαξε γνώμη.
    Απόφυγε να κάνεις περιττά σχόλια και να επαναλάβεις την ίδια φράση κάθε φορά που απαντάς στον χρήστη. Μην απαντάς στον χρήστη με τα προϊόντα που έχει παραγγείλει 
    μέχρι τώρα. Περίμενε να συλλέξεις ολόκληρη την παραγγελία και σημείωσε τα προϊόντα που έχουν παραγγελθεί μέχρι στιγμής, χωρίς να τα επαναλάβεις στον χρήστη. 
    Μην συνεχίζεις τη συνομιλία παριστάνοντας τον χρήστη και μην απαντάς σε ερωτήσεις εκ μέρους του.
    Εάν ο χρήστης ζητήσει, συνδέστε τον με έναν υπάλληλο. Απόφυγε να παρέχεις στην έξοδο ρόλους όπως «άνθρωπος» ή «AI» ή «χρήστης» ή οποιονδήποτε άλλο ρόλο που 
    διευκρινίζει ποιος είπε τι. Απλώς δώσε την απάντησή σου. Δώσε μόνο μία απάντηση και μετά περίμενε να συνεχίσει τη συνομιλία ο χρήστης. Μην επινοείς τη συζήτηση.
    Περίμενε μέχρι να λάβεις μια εισαγωγή από τον χρήστη προτού απαντήσεις σε αυτήν.

    Όταν ο χρήστης ολοκληρώσει την παραγγελία του ή όταν αναφέρει ότι δεν θέλει να προσθέσει άλλα προϊόντα σε αυτήν,
    του ζητάς να πει ακριβώς μία από τις παρακάτω δύο φράσεις «παραλαβή από το κατάστημα» ή «παράδοση σε διεύθυνση». Η πληροφορία αυτή πρέπει να δοθεί από τον χρήστη.
    Αν είναι για παράδοση, ζητάς την διεύθυνση. Αυτή συνήθως δίνεται στην μορφή '<διεύθυνση> <αριθμός>'. Αν δεν δοθεί, συνεχίζεις να ρωτάς μέχρι να δοθεί.
    Αυτή είναι απαραίτητη για να ξέρεις που θα παραδώσουμε την παραγγελία. Σε περίπτωση που δοθεί κάτι στην παραπάνω μορφή, δεν ρωτάς ξανά για διεύθυνση.
    Αν πρόκειται για παραλαβή από το κατάστημα, δεν χρειάζονται περαιτέρω διευκρινίσεις.
    Αφού λάβεις τη διεύθυνση (αν είναι για παράδοση), ζητάς τρόπο πληρωμής (μόνο κάρτα ή μετρητά). Τα στοιχεία της κάρτας (πχ. αριθμός) δεν πρέπει να ζητηθούν.
    Η απάντηση του χρήστη θα πρέπει να είναι είτε «κάρτα» ή «μετρητά».
    Ποτέ μην ζητάς τις ίδιες πληροφορίες δύο φορές. Όταν αυτά τα στοιχεία δοθούν, τα σημειώνεις στην παραγγελία και ευχαριστείς τον χρήστη για την παραγγελία.
    Απόφυγε να δώσεις τη συνολική τιμή της παραγγελίας ή οποιαδήποτε άλλη πληροφορία αφού τον ευχαριστήσεις.
 
	Στο τέλος της πρότασης όπου ευχαριστείς τον χρήστη, θα πρέπει να γράψεις ως τελευταίες λέξεις 
    τη φράση «τέλος παραγγελίας» χωρίς άλλες λέξεις μετά από αυτή. Βεβαιώσου ότι αυτή η φράση έχει γραφτεί μία φορά, ώστε η συνομιλία να σταματήσει.
    Η φράση αυτή πρέπει να γραφτεί μόνο εφόσον οι μέθοδοι πληρωμής και παράδοσης έχουν δοθεί. Αν όχι, συνέχισε να ρωτάς μέχρι να δοθούν.

    Εάν η παραγγελία δεν έχει ολοκληρωθεί ακόμα ή δεν έχουν δοθεί τα στοιχεία παράδοσης ή/και η μέθοδος πληρωμής και αναμένεται μια απάντηση από τον χρήστη,
	συνέχισε τη συνομιλία με τον χρήστη χωρίς να εμφανίσεις την φράση
    «τέλος παραγγελίας». Μην συνεχίσεις τη συνομιλία παριστάνοντας τον χρήστη και μην απαντάς σε ερωτήσεις εκ μέρους του. 
    "
	{chat_history}

	### Χρήστης: {human_input}
	### AI:
	"""

	#Try the following as quick test: '3 souvlakia kotopoulo me sos', '1 patates me sos', '1 coca cola', '1 sprite', '2 (souvlakia) xoirina', '1 pita gyro apo ola',
	# '1 pita souvlaki kotopoulo patata ntomata sos (won't work)', '1 astakomakaronada', '3 podia kamilopardalis'

	# Initialize the LLMChain object
	prompt = PromptTemplate( 
		input_variables=["chat_history", "human_input"], 
		template=template)

	memory = ConversationBufferMemory(memory_key="chat_history") 
		
	llm_chain = LLMChain(
		llm=OpenAI(openai_api_key=openai_api_key,temperature=0, request_timeout=13,max_retries=1,model_name='gpt-3.5-turbo'), #Price ~0.06-0.07$ for basic chat for '-16k'
		#For 4k version, price is 0.03-0.04$ for ~6 user responses - Same price as for -0301 model (will remain until June 2024 at least)
		prompt=prompt, 
		verbose=False, #contains only prompt
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
		no_talk=0 #If no input for 8 seconds connect to employee
		while (query == "None"): #If no input
			query = takecommand() #Take input from user
			no_talk+=1
			if no_talk==4:
				try:
					if 'ευχαριστ' in response.lower(): #If the order has finished but model didn't output 'end of order'
						query="okay τέλος παραγγελίας"
					else:
						flag_employee=1
						break
				except:
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

		text_to_translate = query #Use the Greek address as input to an LLM

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
		
		encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") #Returns the encoding

		# calculate the number of tokens 
		num_token = len(encoding.encode(template+llm_chain.memory.buffer+human_text))
		# print(colored("Number of tokens:",'yellow')+str(num_token))

		if num_token>4000 and flag_model==0:
			flag_model=1
			# print("Now using 16k LLM")
			llm_chain = LLMChain( #with 7 secs will also work
								llm=OpenAI(openai_api_key=openai_api_key,temperature=0, request_timeout=9,max_retries=1,model_name='gpt-3.5-turbo-16k'),
								prompt=prompt, 
								verbose=False, #contains only prompt
								memory=memory)
			response=llm_chain.predict(human_input=human_text) #Predict response using 16k LLM
			
		elif num_token>4000 and flag_model==1:
			response=llm_chain.predict(human_input=human_text) #Predict response using 16k LLM

		else: #Initially get in here
			response=llm_chain.predict(human_input=human_text) #Predict response using LLM - Takes ~1-5.5 secs to run through GPT 3.5 turbo

		# print(llm_chain.memory.buffer)
		print(colored("LLM response:",'yellow')+str(response))

		#Delete any further dialogue output from LLM
		if 'άνθρωπος:' in response.lower():
			print("Human part kept:",response.lower().split('άνθρωπος:'))
			response=response.lower().split('άνθρωπος:')[1]
		if 'chatbot:' in response.lower():
			print("chatbot part kept:",response.lower().split('chatbot:'))
			response=response.lower().split('chatbot:')[1]
		if 'ai:' in response.lower():
			print("AI part kept:",response.lower().split('ai:'))
			response=response.lower().split('ai:')[1]

		if 'υπάλλη' in response.lower(): #If LLM decides that the user should be connected to an employee
			flag_employee=1
			print("After LLM output, connect with employee")
			break #Stop program execution

		response_greek=response
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

		if 'τέλος παραγγ' in response.lower(): #If the order is completed
			flag=1 #Set flag to 1

			summary="Συνόψισε την παρακάτω παραγγελία και επιστρεψέ την ως bullet points. Κάθε bullet point πρέπει να αποτελείται από ένα αντικείμενο. \
				Σημείωσε τυχόν ειδικές παρατηρήσεις, σχόλια ή προτιμήσεις του χρήστη σε bullet point. Μην συνοψίσεις ολόκληρη την παραγγελία σε μια μόνο απάντηση. \
				Σε περίπτωση που ο χρήστης αφαίρεσε ή πρόσθεσε προϊόντα, ανάφερε τις αλλαγές σε ξεχωριστά bullet points. \
				Σημείωσε επίσης τον τρόπο πληρωμής και εάν πρόκειται για παραλαβή από το κατάστημα ή παράδοση, μαζί με τη διεύθυνση (εάν παρέχεται). \
				Θα πρέπει επίσης να σημειώσεις εάν ο χρήστης χρησιμοποίησε ακατάλληλες λέξεις ή είπε κάτι που συνήθως δεν λέγεται όταν κάποιος παραγγέλνει φαγητό. \
				Εάν όχι, τότε μην αναφέρεις αυτό το σημείο. Ξεκίνησε κατευθείαν την απάντησή σου με τα bullet points σαν να μην έπρεπε να απαντήσεις στον χρήστη. \
				Πρόσεξε να μην θεωρήσεις ως ξεχωριστά προϊόντα τα συστατικά που πρέπει να περιέχονται σε ένα συγκεκριμένο προϊόν. \
				Η συνομιλία από την παραγγελία είναι η εξής: "

			order_chat=llm_chain.memory.buffer

			llm=OpenAI(openai_api_key=openai_api_key,temperature=0,request_timeout=20,max_retries=1,model_name='gpt-3.5-turbo')
			response_final=llm.predict(summary+order_chat) #Predict response using LLM - Takes ~1-5.5 secs to run through GPT 3.5 turbo
			print(colored("LLM response - order summary:",'yellow')+str(response_final))
			print('\n')

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

			response_greek_final=response_final

			########################################
			#Menu similarity search

			# template_menu="""Για τα είδη που παρήγγειλε ο χρήστης δώσε την τιμή τους από το μενού. Εάν το προϊόν δεν είναι διαθέσιμο στο μενού, 
			# αναφερέ το ρητά στην έξοδο. Εάν δεν υπάρχει προϊόν στην περιγραφή, αναφέρέ ρητά στην έξοδο ότι δεν υπάρχει κανένα προϊόν. 
			# Μην παρέχεις τιμές αν δεν υπάρχουν στο μενού. Δώσε τιμή μόνο αν συσχετίζεται με το αντικείμενο της παραγγελίας. Μην επινοείς τιμές.
			# Τα είδη που παραγγέλνει ο χρήστης πρέπει να σχετίζονται με τρόφιμα. Αν δεν σχετίζονται απλά πες ότι δεν είναι διαθέσιμα. 
			# Απάντησε αναφέροντας το ακριβές είδος που πήρε ο χρήστης, το αντικείμενο το μενού που του μοιάζει περισσότερο και την τιμή του (αν υπάρχουν).
			# Προσπάθησε να αντιστοιχίσεις μόνο το αντικείμενο/φράση που δίνεται με ένα αντικείμενο του μενού. 
			# {context}

			# {chat_history}

			# # Χρήστης: {human_input}
			# # Chatbot:"""

			# prompt_menu = PromptTemplate(input_variables=["chat_history", "human_input","context"], template=template_menu)  

			# memory_menu = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

			# #This method uses all text in menu
			# llm_chain_menu = load_qa_chain(OpenAI(openai_api_key=openai_api_key,temperature=0, request_timeout=9,max_retries=1, model_name='gpt-3.5-turbo'), 
			# 		chain_type="stuff", memory=memory_menu, prompt=prompt_menu) #'stuff' to use all text in menu as input to LLM
			# # Another approach using RetrievalQA https://python.langchain.com/docs/modules/chains/popular/vector_db_qa
			# # This method is the same as above. There would only be a difference if we haven't used the results of the similarity search above as input to the load_qa_chain.

			# print("Now checking items in the following list:",response_greek_final.split('\n'))

			# if 'coca-cola' in response_greek_final.lower():
			# 	response_greek_final=response_greek_final.replace('coca-cola','coca cola')

			# #For each item ordered by the user, perform similarity search and let the LLM decide the total price
			# all_items=[]
			# all_items_greek=[]

			# if len(response_greek_final.split('\n'))<2: #In case all items returned in one line
			# 	response_greek_final=response_greek_final.split('-')
			# 	response_greek_final='\n'.join(response_greek_final)

			# for item in response_greek_final.split('\n'):
			# 	if 'παράδοση' in item.lower() or 'πληρωμ' in item.lower() or 'παραγγ' in item or item=='\n' or item=='': #These are not items
			# 	#above changed since last item could be in the same line as 'παρατ' - or 'παρατ' in item
			# 	# if item=='\n' or item=='':
			# 		pass
			# 	else:
			# 		print("Item is",item)
					
			# 		if 'χοιρινά' or 'χοιρινα' in item.lower() and ('μπριζολ' not in item.lower() and 'σουβλ' not in item.lower()):
			# 			item = item.lower().replace('χοιρινά','σουβλάκια χοιρινά')
			# 			item = item.lower().replace('χοιρινα','σουβλάκια χοιρινά')

			# 		print("List of similar items from menu:",docsearch.similarity_search(item,k=1)) #If user added 2 items in a second prompt, use k=2 and 16k model
			# 		final_doc=docsearch.similarity_search(item,k=1)
			# 		out=llm_chain_menu.run(input_documents=final_doc, human_input=item,  return_only_outputs=True, verbose=True) 
			# 		#Run the chain - Response time ranges from 3 to 7 seconds

			# 		print("LLM menu search result:",out)
			# 		print('\n')
			# 		all_items.append(out) #Append all items to a list to get total price of order below

			# 		response_greek_final_items = out
			# 		# print("Translation to Greek:",response_greek_final_items) #Print the final order summary
			# 		all_items_greek.append(response_greek_final_items)

			# all_items_greek_string='\n'.join(all_items_greek)
			# print(colored("Greek order with menu check:",'blue')+str(all_items_greek_string))

			# all_items_string=' '.join(all_items)

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
# OK - 'pita souvlaki kotopoulo patata ntomata sos' doesn't work since it thinks that the sauce consists of potatoes and ntomatoes
# OK - 'pita giro apo ola' will translate to 'στρογγυλη πιτα απο ολα' and won't be found in menu.
# 3) For price estimation, use agents? For example, if two same items are ordered (2 souvlakia xoirina), will it find the correct price? => Yes, so no need for agents
# 4) Add a general try except where we connect to an employee in case of any error (eg. OpenAI call failed, no sound available to speak etc.) => Done
# 5) Perform similarity search if user asks 'do you have souvlakia today?' - Can't be done since the speech recognition module doesn't recognize questions.
# 6) If context length exceeded, only then transfer to 16k model - Done

# 7) Set a maximum duration of conversation (to ensure we are not getting charged for fun) and then connect to employee, informing them of the duration. => Not done
# 8) Create prompts with examples eg. user ordered 2 xoirina, 3 kotopoulo then removed 2 kototpoulo so final order should be... => Not done