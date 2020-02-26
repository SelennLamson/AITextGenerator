#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import random
from io import BytesIO
from typing import List, Dict, Tuple

from src.flexible_models import *
from src.utils import *


PORT = 7777
Handler = http.server.SimpleHTTPRequestHandler


class Generator:
	def __init__(self):
		print("Loading models...")
		self.ner_model = FlexibleBERTNER(BERT_NER_LARGE, 1, 2000)
		print("Models loaded, ready to serve!")

	def perform_ner(self, text) -> Dict[str, Tuple[str, float]]:
		return self.ner_model([text])[0]


class AITextGeneratorHTTPServer(Handler):
	def do_OPTIONS(self):
		self.send_response(200, "ok")
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
		self.send_header("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Headers, X-Requested-With")
		self.end_headers()

	# def do_GET(self):
	# 	self.send_response(200)
	# 	self.send_header("Access-Control-Allow-Origin", "*")
	# 	self.end_headers()
	#
	# 	data = json.load(open('../../data/preproc/517_preproc.json', 'r'))
	# 	paragraphs = data['novel']['paragraphs']
	# 	p = random.choice(paragraphs)['text']
	#
	# 	response = BytesIO()
	# 	response.write(bytes(p, 'utf-8'))
	# 	self.wfile.write(response.getvalue())

	def do_POST(self):
		self.send_response(200)
		self.send_header("Access-Control-Allow-Origin", "*")
		self.end_headers()

		content_length = int(self.headers['Content-Length'])
		body = str(self.rfile.read(content_length))

		body = ' '.join(body.split('<p>')[1:])
		body = ' '.join(body.split('</p>')[:-1])
		body = body.replace('\\', '')

		ent_dict = generator.perform_ner(body)
		entities = [v[0] + ':' + k for k, v in ent_dict.items()]

		response = BytesIO()
		for i, e in enumerate(entities):
			response.write(bytes(e + ('</p><p>' if i < len(entities) - 1 else ''), 'utf-8'))
		self.wfile.write(response.getvalue())


with socketserver.TCPServer(("0.0.0.0", PORT), AITextGeneratorHTTPServer) as httpd:
	generator = Generator()
	print("serving at port", PORT)
	httpd.serve_forever()

