Here are the data-files after preprocessing, entity recognition and summarization.

**Format:** JSON

**Structure:**
```json
{
  "novel": {
    "title": "The Lord of The Rings",
    "author": "J. R. R. Tolkien",
    "theme": "Fantasy",
    "paragraphs": [
      {
        "persons": ["Frodon", "Gandalf"],
        "locations": ["Shire"],
        "organisations": [],
        "misc": [],
        "size": 212,
        "summaries": [
          "Gandalf visits Frodon.",
          "Gandalf enters Frodon's house."
        ],
        "text": "Gandalf entered the small house Frodon was living in, typical of the Shire. [...]"
      },
      {
        "persons": [],
        "locations": [],
        "organisations": [],
        "misc": ["Hobbit"],
        "size": 245,
        "summaries": [
          "He goes into the kitchen.",
          "He enters the kicthen."
        ],
        "text": "Without warning, he settled down in the Hobbit's kitchen. [...]"
      }
    ]
  }
}
```
