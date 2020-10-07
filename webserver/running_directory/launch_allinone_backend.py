#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../../')

from webserver.webutils import CONFIG, launch_backend
launch_backend(CONFIG['master-port'])
