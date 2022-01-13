# A quantum circuit simulation library

The code is already packaged. First create and activate a virtualenv :

```bash
virtualenv venv
source venv/bin/activate
```

Then install the package:

```bash
python setup.py install
```

You should then be able to run the tests (that will fail, all good):

```bash
python -m pytest tests/testing.py
```

# Development

We will develop (at least) three simulators, each in its own file (under `src/qps`)
