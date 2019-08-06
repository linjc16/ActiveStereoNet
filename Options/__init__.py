import json
import os

from collections import OrderedDict
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%Y%m%d')

def parse_opt(opt_path):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()

    return opt
