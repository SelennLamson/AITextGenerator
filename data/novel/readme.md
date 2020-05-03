Contains each book's full text and metadata. 
Obtained from 'metadata/files/' by additionally detecting entities and storing their position in the text.


**Format:** JSON

**Structure:**
```json
{
     "author": [
         "Austen, Jane"
     ],
     "title": [
         "Pride and Prejudice"
     ],
     "language": [
        "en"
     ],
     "theme": [
        "Social classes -- Fiction",
        "Young women -- Fiction",
        ...
     ],
     "id": "1342",
     "genre": [
        "romance"
     ],
    "text": "However little known the feelings or views of such a man may be [...]",
    "persons": {
        "130": "Jane Austen",
        "2079": "Long",
        "2653": "Morris",
        ...
    },
    "locations": {
        "1958": "Netherfield Park",
        "2423": "Netherfield",
        "2500": "England",
        ...
    },
    "organisations": {
        "117298": "WILLIAM",
        "154852": "Pemberley House",
        "441638": "Longbourn",
        ...
    },
    "misc": {
        "100": "Pride and Prejudice",
        "2716": "Michaelmas",
        "30446": "Meryton",
        ...
    }
}
```
