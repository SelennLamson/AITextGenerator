#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../../')

from webserver.webutils import ORDERS, CONFIG, launch_backend

ORDERS.remove('ipconfig')
ORDERS.remove('feedback')
ORDERS.remove('share')
ORDERS.remove('getshared')
ORDERS.remove('extract_entities')
launch_backend(CONFIG['generation-port'])
