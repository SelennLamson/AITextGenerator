#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import random
import time
from io import BytesIO
from typing import List, Dict, Tuple

from src.utils import *
from src.flexible_models.flexible_GPT2 import FlexibleGPT2
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER
from src.model_use.text_generation import TextGeneration
from src.torch_loader.vectorize_input import GenerationInput

from transformers import GPT2LMHeadModel, GPT2Tokenizer


PORT = 7777
Handler = http.server.SimpleHTTPRequestHandler


class Generator:
    def __init__(self):
        print("Loading models...")

        self.ner_model = FlexibleBERTNER(BERT_NER_LARGE, 32, 128)

        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_flexible_model = FlexibleGPT2(gpt2_model, gpt2_tokenizer, DEFAULT_DECODING_STRATEGY)
        self.gen_model = TextGeneration(gpt2_flexible_model)

        print("Models loaded, ready to serve!")

    def perform_ner(self, text) -> Dict[str, Tuple[str, float]]:
        return self.ner_model([text], verbose=0)[0]

    def generate_text(self, p1, sp2, p3, entities, genre, size):
        context_input = GenerationInput(P1=p1,
                                        P3=p3,
                                        genre=[genre],
                                        entities=entities,
                                        size=size,
                                        summary=sp2)
        predictions = self.gen_model(context_input, nb_samples=4)
        return predictions


class AITextGeneratorHTTPServer(Handler):
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

        content_length = int(self.headers['Content-Length'])
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

        response = BytesIO()

        try:
            params = json.loads(cleaned)
            order = params["order"]

            if order == 'extract_entities':
                ent_dict = generator.perform_ner(params['body'])
                entities = set([v[0] + ':' + k.strip() for k, v in ent_dict.items()])

                response.write(bytes('</p><p>'.join(entities), 'utf-8'))

            elif order == 'generate':
                p1 = params["p1"]
                sp2 = params["sp2"]
                p3 = params["p3"]
                entities = params["entities"]
                theme = params["theme"]
                size = params["size"]

                generated = generator.generate_text(p1, sp2, p3, entities, theme, size)

                response.write(bytes('||'.join(generated), 'utf-8'))

        except (KeyError, json.JSONDecodeError) as e:
            if isinstance(e, KeyError):
                print("KeyError while parsing JSON")
            else:
                print("JSON parsing error")
                print(e)

        self.wfile.write(response.getvalue())


with socketserver.TCPServer(("0.0.0.0", PORT), AITextGeneratorHTTPServer) as httpd:
    generator = Generator()
    print("serving at port", PORT)
    httpd.serve_forever()

