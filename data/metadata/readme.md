clean_data.json contains the raw metadata content. 
It is used to create a json file per book stored in the 'files' folder,
whose structure is detailed below. 
We constructed the 'genre' ourselves, retrieved the text from Gutenberg, cleaned it
and finally filtered the books based on the language and the genre (keep only those we are interested in - english & novels)




**Format:** JSON

**Structure:**
```json
{
 "author": [
  "Wells, H. G. (Herbert George)"
 ],
 "title": [
  "When the Sleeper Wakes"
 ],
 "language": [
  "en"
 ],
 "theme": [
  "Twenty-first century -- Fiction",
  "London (England) -- Fiction",
  "Dystopias -- Fiction",
  "Time travel -- Fiction",
  "Technological innovations -- Fiction",
  "PR",
  "Science fiction"
 ],
 "id": "775",
 "genre": [
  "science-fiction"
 ],
 "text": "blablabla"
}
```


