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
from webserver.generation_backend import Generator
from webserver.ner_backend import Extractor

CONFIG = json.load(open('../config.json', 'r'))
Handler = http.server.SimpleHTTPRequestHandler
WEBSERVICE_FEEDBACK = CONFIG['webservice-data-path']
WEBSERVICE_SHARED = WEBSERVICE_FEEDBACK + 'shared.json'

GENERATOR = Generator()
EXTRACTOR = Extractor()


class AllInOneBackendHTTPServer(Handler):
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

				if order == 'ipconfig':
					rep = CONFIG['master-ip'] + ':' + CONFIG['master-port'] + '|' + CONFIG['master-ip'] + ':' + CONFIG['master-port']
					response.write(bytes(rep, 'utf-8'))

				elif order == 'feedback':
					mail = params['mail']
					message = params['message']

					filename = WEBSERVICE_FEEDBACK + datetime.datetime.now().strftime("%Y-%m-%d") + '_feedbacks.json'
					if os.path.exists(filename):
						json_feedbacks = json.load(open(filename, 'r', encoding='utf-8'))
					else:
						json_feedbacks = []

					json_feedbacks.append({'mail': mail, 'message': message, 'time': datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")})
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

					response.write(bytes('||'.join(messages[:100]), 'utf-8'))

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
					if os.path.exists(WEBSERVICE_FEEDBACK + 'record.json'):
						saved = json.load(open(WEBSERVICE_FEEDBACK + 'record.json', 'r', encoding='utf-8'))
					else:
						saved = []
					saved.append(params)
					json.dump(saved, open(WEBSERVICE_FEEDBACK + 'record.json', 'w', encoding='utf-8'))

					response.write(bytes('||'.join(generated), 'utf-8'))

				elif order == 'extract_entities':
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


def launch_all_in_one_backend():
	with socketserver.TCPServer(("0.0.0.0", int(CONFIG['master-port'])), AllInOneBackendHTTPServer) as httpd:
		print("serving at port", CONFIG['master-port'])
		httpd.serve_forever()

