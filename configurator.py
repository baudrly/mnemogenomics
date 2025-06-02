"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--'), f"{arg}"
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            # Allow type change for string to bool/int if that's the intent
            # print(f"Key: {key}, Current Type: {type(globals()[key])}, Attempted Type: {type(attempt)}")
            if isinstance(globals()[key], (bool, int, float)) and isinstance(attempt, (bool, int, float, str)):
                 try:
                    # If current is bool, try to convert string 'True'/'False'
                    if isinstance(globals()[key], bool) and isinstance(attempt, str):
                        if attempt.lower() == 'true': attempt = True
                        elif attempt.lower() == 'false': attempt = False
                        else: pass # Keep as string if not 'true'/'false' to fail type assertion later
                    # If current is int/float, try to convert string digit
                    elif isinstance(globals()[key], (int,float)) and isinstance(attempt, str):
                        pass # literal_eval already handled this, if it failed, then it's a string not meant to be number
                 except ValueError:
                     pass # keep `attempt` as is, type check will fail if types incompatible
            
            assert type(attempt) == type(globals()[key]) or \
                   (isinstance(globals()[key], float) and isinstance(attempt, int)) or \
                   (isinstance(globals()[key], int) and isinstance(attempt, float)), \
                   f"Type mismatch for key {key}: expected {type(globals()[key])} but got {type(attempt)} for value {val}"
            
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")