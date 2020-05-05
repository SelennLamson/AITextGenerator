Final data format, ready to be used. 
Builds on 'data/novel/' and 'data/summaries/'. After NER, the text was split into paragraphs,
each of which was then summarized using four different summarizers.  

**Format:** JSON

**Structure:**
```json
{
 "author": [
  "Douglass, Frederick"
 ],
 "title": [
  "Narrative of the Life of Frederick Douglass, an American Slave"
 ],
 "language": [
  "en"
 ],
 "theme": [
  "African American abolitionists -- Biography",
  "Douglass, Frederick, 1818-1895",
  "E300",
  "Slaves -- United States -- Biography",
  "Abolitionists -- United States -- Biography"
 ],
 "id": "23",
 "genre": [
  "biography/history"
 ],
 "paragraphs": [
  {
   "size": 1460,
   "text": "After much deliberation, however, he consented to make a trial; and ever since that period, he has acted as a lecturing agent, under the auspices either of the American or the Massachusetts Anti-Slavery Society. In labors he has been most abundant; and his success in combating prejudice, in gaining proselytes, in agitating the public mind, has far surpassed the most sanguine expectations that were raised at the commencement of his brilliant career. He has borne himself with gentleness and meekness, yet with true manliness of character. As a public speaker, he excels in pathos, wit, comparison, imitation, strength of reasoning, and fluency of language. There is in him that union of head and heart, which is indispensable to an enlightenment of the heads and a winning of the hearts of others. May his strength continue to be equal to his day! May he continue to \"grow in grace, and in the knowledge of God,\" that he may be increasingly serviceable in the cause of bleeding humanity, whether at home or abroad! It is certainly a very remarkable fact, that one of the most efficient advocates of the slave population, now before the public, is a fugitive slave, in the person of _Frederick Douglass_; and that the free colored population of the United States are as ably represented by one of their own number, in the person of _Charles Lenox Remond_, whose eloquent appeals have extorted the highest applause of multitudes on both sides of the Atlantic.",
   "summaries": {
    "T5": ", he has borne himself with gentleness and meekness. He excels in pathoses; his ability to combat",
    "BART": "After much deliberation, however, he consented to make a trial; and ever since that period, he has acted as a lecturing agent. As a public speaker, he excels in pathos, wit, comparison, imitation, strength of reasoning, and fluency of language.",
    "PYSUM": "After much deliberation, however, he consented to make a trial; and ever since that period, he has acted as a lecturing agent, under the auspices either of the American or the Massachusetts Anti-Slavery Society.",
    "KW": "public - lenox - anti - slave - colored - wit - population - imitation strength"
   },
   "persons": [],
   "organisations": [],
   "locations": [
    "United States",
    "Charles"
   ],
   "misc": [
    "American"
   ]
  }, 
...
}
```
