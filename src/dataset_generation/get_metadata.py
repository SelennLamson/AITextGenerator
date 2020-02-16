
###########################################
# Set up 
###########################################
#pip install gutenberg
#git clone https://github.com/c-w/Gutenberg.github
#cd Gutenberg
#pip install r-requirements-dev.pip
#brew install berkeley-db4
#pip install .

# Import libraries 
from gutenberg.acquire import load_etext, get_metadata_cache
from gutenberg.cleanup import strip_headers 
from gutenberg.query import get_etexts, list_supported_metadatas, get_metadata

# Check what kind of metadata we can get
print(list_supported_metadatas()) # prints (u'author', u'formaturi', u'language', ...)

# Download a text 
text = strip_headers(load_etext(2701)).strip()
print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'

# Populate the metadata cache
cache = get_metadata_cache()
cache.populate()

# Extract metadata from that file 
print(get_metadata('title', 2701))  # prints frozenset([u'Moby Dick; Or, The Whale'])
print(get_metadata('author', 2701)) # prints frozenset([u'Melville, Hermann'])

print(get_etexts('title', 'Moby Dick; Or, The Whale'))  # prints frozenset([2701, ...])
print(get_etexts('author', 'Melville, Hermann'))        # prints frozenset([2701, ...])

"""
# Cache with more control 
from gutenberg.acquire.metadata import SqliteMetadataCache
cache = SqliteMetadataCache('/my/custom/location/cache.sqlite')
cache.populate()
set_metadata_cache(cache)
"""