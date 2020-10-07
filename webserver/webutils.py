#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import random
import time
import datetime
import os
from io import BytesIO
from typing import List, Dict, Tuple
from transformers import GPT2Tokenizer

from src.utils import *
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.torch_loader.vectorize_input import GenerationInput
from src.torch_loader import VectorizeParagraph, VectorizeMode
from src.flexible_models.GPT2_lm_segment_model import GPT2LMSegmentModel

from webserver.webutils import *

ORDERS = {'ipconfig', 'feedback', 'share', 'getshared', 'generate', 'extract_entities'}
CONFIG = json.load(open('config.json', 'r'))
WEBSERVICE_FEEDBACK = CONFIG['webservice-data-path']
WEBSERVICE_SHARED = WEBSERVICE_FEEDBACK + 'shared.json'
WEBSERVICE_RECORD = WEBSERVICE_FEEDBACK + 'record.json'


class Generator:
	def __init__(self):
		self.gpt2_model = None
		self.gpt2_tokenizer = None
		self.gpt2_flexible_model = None
		self.vectorizer = None

		self.ready = False

	def setup(self):
		print("Loading model...")

		self.gpt2_model = GPT2LMSegmentModel.from_pretrained(CONFIG['generation-model-path'])
		self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(CONFIG['generation-model-path'])
		self.gpt2_flexible_model = FlexibleGPT2(self.gpt2_model, self.gpt2_tokenizer, DEFAULT_DECODING_STRATEGY)
		self.vectorizer = VectorizeParagraph(tokenizer=self.gpt2_tokenizer,
                                             block_size=GPT2_BLOCK_SIZE,
                                             mode=VectorizeMode.GENERATE,
                                             use_context=True)

		self.ready = True

		print("Model loaded, ready to serve!")

	def generate_text(self, p1, sp2, p3, persons, locations, organisations, misc, genre, size):
		context_input = GenerationInput(P1=p1,
										P3=p3,
										genre=[genre],
										persons=persons,
										locations=locations,
										organisations=organisations,
										misc=misc,
										size=size,
										summary=sp2)
		input_ids, _ = self.vectorizer(context_input)
		return self.gpt2_flexible_model.predict(input_ids, nb_samples=4)
GENERATOR = Generator()


class Extractor:
	def __init__(self):
		self.ner_model = None
		self.ready = False

	def setup(self):
		print("Loading model...")

		self.ner_model = FlexibleBERTNER(CONFIG['ner-model-path'], 32, 128)
		self.ready = True

		print("Model loaded, ready to serve!")

	def perform_ner(self, text) -> Dict[str, Tuple[str, float]]:
		return self.ner_model([text], verbose=0)[0]
EXTRACTOR = Extractor()


def clean(txt):
	cleaned = ""
	in_chevrons = False
	for c in txt:
		if in_chevrons:
			if c == '>':
				in_chevrons = False
		else:
			if c == '<':
				in_chevrons = True
			else:
				cleaned += c
	return cleaned


def handle_request(headers, file):

	content_length = int(headers['Content-Length'])
	if content_length > 20000:
		return 'ERROR'


	content = file.read(content_length).decode("utf-8").replace('</p>', '\\n')
	try:
		params = json.loads(content)
		order = params['order']

		for key, value in params.items():
			params[key] = clean(value)
	except (KeyError, json.JSONDecodeError) as e:
		if isinstance(e, KeyError):
			print("KeyError while parsing JSON")
		else:
			print("JSON parsing error")
			print(e)
		return 'ERROR'


	if order not in ORDERS:
		return 'ERROR'


	if order == 'ipconfig':
		rep = random.choice(CONFIG['generation-ips']) + ':' + CONFIG['generation-port'] + '|' + CONFIG['ner-ip'] + ':' + \
			  CONFIG['ner-port']
		return rep


	elif order == 'feedback':
		mail = params['mail']
		message = params['message']

		filename = WEBSERVICE_FEEDBACK + datetime.datetime.now().strftime("%Y-%m-%d") + '_feedbacks.json'
		if os.path.exists(filename):
			json_feedbacks = json.load(open(filename, 'r', encoding='utf-8'))
		else:
			json_feedbacks = []

		json_feedbacks.append(
			{'mail': mail, 'message': message, 'time': datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")})
		json.dump(json_feedbacks, open(filename, 'w', encoding='utf-8'))


	elif order == 'share':
		pseudo = params['pseudo'].replace('|', '')
		text = params['text'].replace('|', '')

		if os.path.exists(WEBSERVICE_SHARED):
			json_shared = json.load(open(WEBSERVICE_SHARED, 'r', encoding='utf-8'))
		else:
			json_shared = []

		json_shared.append(
			{'pseudo': pseudo, 'text': text, 'time': datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")})
		json.dump(json_shared, open(WEBSERVICE_SHARED, 'w', encoding='utf-8'))


	elif order == 'getshared':
		if os.path.exists(WEBSERVICE_SHARED):
			json_shared = json.load(open(WEBSERVICE_SHARED, 'r', encoding='utf-8'))
		else:
			json_shared = []

		messages = []
		for shared in json_shared:
			pseudo = shared['pseudo']
			text = shared['text']
			message_time = shared['time']
			messages.append(text + '|' + pseudo + '|' + message_time)

		messages.sort(key=lambda x: x.split('|')[2], reverse=True)
		return '||'.join(messages[:100])


	elif order == 'generate':
		p1 = params["p1"]
		sp2 = params["sp2"]
		p3 = params["p3"]
		persons = params["persons"]
		locations = params["locations"]
		organisations = params["organisations"]
		misc = params["misc"]
		genre = params["genre"]
		size_tok = params["size"]

		size = SMALL
		for s in SIZES:
			if s.token == size_tok:
				size = s

		if not GENERATOR.ready:
			GENERATOR.setup()
		generated = GENERATOR.generate_text(p1, sp2, p3, persons, locations, organisations, misc, genre, size)

		params['generated'] = generated
		if os.path.exists(WEBSERVICE_DATA + 'record.json'):
			saved = json.load(open(WEBSERVICE_DATA + 'record.json', 'r', encoding='utf-8'))
		else:
			saved = []
		saved.append(params)
		json.dump(saved, open(WEBSERVICE_DATA + 'record.json', 'w', encoding='utf-8'))

		response.write(bytes('||'.join(generated), 'utf-8'))


	elif order == 'extract_entities':
		if not EXTRACTOR.ready:
			EXTRACTOR.setup()
		ent_dict = EXTRACTOR.perform_ner(params['body'])
		entities = set([v[0] + ':' + k.strip() for k, v in ent_dict.items()])

		response.write(bytes('</p><p>'.join(entities), 'utf-8'))


	else:
		return 'ERROR'


Handler = http.server.SimpleHTTPRequestHandler
class BackendHTTPServer(Handler):
	def do_OPTIONS(self):
		self.send_response(200, "ok")
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
		self.send_header("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, X-Requested-With")
		self.end_headers()

	def do_POST(self):
		self.send_response(200)
		self.send_header("Access-Control-Allow-Origin", "*")
		self.end_headers()

		response = BytesIO()
		response.write(bytes(handle_request(self.headers, self.rfile), 'utf-8'))

		self.wfile.write(response.getvalue())


def launch_backend(port):
	with socketserver.TCPServer(("0.0.0.0", int(port)), BackendHTTPServer) as httpd:
		print("Serving at port", port)
		httpd.serve_forever()

