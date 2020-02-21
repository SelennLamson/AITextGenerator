import http.server
import socketserver
import json
import random
from io import BytesIO

from src.third_party.BERT_NER.bert import Ner
from src.dataset_generation.ent_sum_preprocessing import *

PORT = 7777
Handler = http.server.SimpleHTTPRequestHandler


class Generator:
	def __init__(self):
		print("Loading models...")
		self.model = Ner("../../models/entity_recognition/BERT_NER_Large/")
		print("Models loaded, ready to serve!")
generator = Generator()


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

		ent_dict = perform_ner(generator.model, body, 2000)
		entities = [v[0] + ': ' + k for k, v in ent_dict.items()]

		response = BytesIO()
		for i, e in enumerate(entities):
			response.write(bytes(e + ('</p><p>' if i < len(entities) - 1 else ''), 'utf-8'))
		self.wfile.write(response.getvalue())


with socketserver.TCPServer(("0.0.0.0", PORT), AITextGeneratorHTTPServer) as httpd:
	print("serving at port", PORT)
	httpd.serve_forever()


