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

from src.utils import *
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.torch_loader.vectorize_input import GenerationInput
from src.torch_loader import VectorizeParagraph, VectorizeMode
from src.flexible_models.GPT2_lm_segment_model import GPT2LMSegmentModel

from transformers import GPT2Tokenizer


CONFIG = json.load(open('../config.json', 'r'))
Handler = http.server.SimpleHTTPRequestHandler
WEBSERVICE_DATA = CONFIG['webservice-data-path']


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

class GenerationBackendHTTPServer(Handler):
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
		content_length = int(self.headers['Content-Length'])

		if content_length <= 20000:
			content = self.rfile.read(content_length).decode("utf-8").replace('</p>', '\\n')

			cleaned = ""
			in_chevrons = False
			for c in content:
				if in_chevrons:
					if c == '>':
						in_chevrons = False
				else:
					if c == '<':
						in_chevrons = True
					else:
						cleaned += c

			try:
				params = json.loads(cleaned)
				order = params["order"]

				if order == 'generate':
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

				else:
					response.write(bytes('ERROR', 'utf-8'))

			except (KeyError, json.JSONDecodeError) as e:
				response.write(bytes('ERROR', 'utf-8'))
				if isinstance(e, KeyError):
					print("KeyError while parsing JSON")
				else:
					print("JSON parsing error")
					print(e)

		self.wfile.write(response.getvalue())


def launch_generation_backend():
	with socketserver.TCPServer(("0.0.0.0", int(CONFIG['generation-port'])), GenerationBackendHTTPServer) as httpd:
		print("serving at port", CONFIG['generation-port'])
		httpd.serve_forever()



