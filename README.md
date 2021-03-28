Fast Word Mover's Distance [![Build Status](https://travis-ci.com/src-d/wmd-relax.svg?branch=master)](https://travis-ci.com/src-d/wmd-relax) [![PyPI](https://img.shields.io/pypi/v/wmd.svg)](https://pypi.python.org/pypi/wmd) [![codecov](https://codecov.io/github/src-d/wmd-relax/coverage.svg)](https://codecov.io/gh/src-d/wmd-relax)
==========================

Calculates Word Mover's Distance as described in
[From Word Embeddings To Document Distances](http://www.cs.cornell.edu/~kilian/papers/wmd_metric.pdf)
by Matt Kusner, Yu Sun, Nicholas Kolkin and Kilian Weinberger. 

Modified by Thais to work using concatenation of word embeddings.

<img src="doc/wmd.png" alt="Word Mover's Distance" width="200"/>

The high level logic is written in Python, the low level functions related to
linear programming are offloaded to the bundled native extension. The native
extension can be built as a generic shared library not related to Python at all.
**Python 2.7 and older are not supported.** The heavy-lifting is done by
[google/or-tools](https://github.com/google/or-tools).


### Installation

```
pip3 install git+git://github.com/ThaisLuca/wmd-relax
```
Tested on Linux and macOS.

### Usage

You should have the embeddings numpy array and the nbow model - that is,
every sample is a weighted set of items, and every item is embedded.

```python
import numpy as np
import spacy
import wmd

doc1 = 'company has office'
doc2 = 'team plays for league'

# Load your pre-trained SpaCy model
nlp = spacy.blank("en").from_disk('spacy')

# Instanciate SpacySimilarityHook using the pre-trained model
wmd_instance = wmd.WMD.SpacySimilarityHook(nlp)

# Generate word embeddings using concatenation to obtain single vectors
embeddings = [np.concatenate([nlp.vocab[w].vector for w in doc1.split()]), 
np.concatenate([nlp.vocab[w].vector for w in doc2.split()])]

# Fix document 1 to be the same size as document 2 (as len(doc2) > len(doc1)). Fill it up with zeros.
dimension = 300 # vectors length 
dim = dimension * ((len(embeddings[1]) - len(embeddings[0]))//dimension)
embeddings[0], embeddings[1] =  np.concatenate((embeddings[0], np.zeros(dim))), embeddings[1]

# Calculate similarity
print(wmd_instance.compute_similarity(nlp("companyhasoffice"), nlp("teamplaysforleague"), evec=np.array(embeddings, dtype=np.float32), single_vector=True))
```

### Tests

Tests are in `test.py` and use the stock `unittest` package.

### Documentation

```
cd doc
make html
```

The files are in `doc/doxyhtml` and `doc/html` directories.

### Contributions

...are welcome! See [CONTRIBUTING](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

### License
[Apache 2.0](LICENSE.md)

#### README {#ignore_this_doxygen_anchor}
