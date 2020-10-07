#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../../')

from webserver.webutils import ORDERS, CONFIG, launch_backend

ORDERS.remove('generate')
ORDERS.remove('extract_entities')
launch_backend(CONFIG['master-port'])
