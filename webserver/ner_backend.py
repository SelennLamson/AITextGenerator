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
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER


CONFIG = json.load(open('webserver/config.json', 'r'))
Handler = http.server.SimpleHTTPRequestHandler


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


class NerBackendHTTPServer(Handler):
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

				if order == 'extract_entities':

					if not EXTRACTOR.ready:
						EXTRACTOR.setup()
					ent_dict = EXTRACTOR.perform_ner(params['body'])
					entities = set([v[0] + ':' + k.strip() for k, v in ent_dict.items()])

					response.write(bytes('</p><p>'.join(entities), 'utf-8'))

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


def launch_ner_backend():
	with socketserver.TCPServer(("0.0.0.0", int(CONFIG['ner-port'])), NerBackendHTTPServer) as httpd:
		print("serving at port", CONFIG['ner-port'])
		httpd.serve_forever()


