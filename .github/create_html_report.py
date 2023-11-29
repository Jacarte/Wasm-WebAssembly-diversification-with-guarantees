import json
from jinja2 import PackageLoader, select_autoescape
from jinja2.nativetypes import NativeEnvironment
import sys
import os

DIR = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':

    data = sys.argv[1]
    out = sys.argv[2]

    env = NativeEnvironment()
    env.filters['zip'] = zip
    
    template = env.from_string(open(f"{DIR}/templates/matches.html", 'r').read())

    data = json.loads(open(data, 'r').read())
    rendered = template.render(data=data, counts=data['counts'])

    open(out, 'w').write(rendered)
