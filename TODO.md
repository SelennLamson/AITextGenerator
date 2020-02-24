# TODO-LIST

## Data acquisition
1. Download data `--> data/raw/ID.txt`
2. Verify integrity (english, consistensy) for at least 100 novels
3. **DONE** - Select ~10 caracteristic novels with their themes (by hand)
4. **DONE** - Script to extract clean novels and their themes (at scale): catalog file with file/title/author/theme + novel text files

`--> ID_novel.json`

## Preprocessing
0. **DONE** - Define preproc_JSON architecture
1. **DONE** - Paragraph separator (parameters: min size, max size) - V1
2. **DONE** - Study paragraph size distribution: is it consistent?
3. **DONE** - Paragraph adaptive separator - V2
4. Check extraction consistensy

`--> ID_preproc.json`

## NER & Summarization
0. **DONE** - Define ent_sum.JSON architecture
1. **DONE** - BERT-NER script architecture `--> add to JSON`
2. Research: Are other NER algorithms working better?
3. Summarizers: apply different summarizers on each paragraph `--> add to JSON`

`--> ID_entsum.json`

## Evaluation STEP 0
1. Evaluate summarization quality by hand
2. Evaluate BERT distance between P2 and SP2, with real summaries dataset as a benchmark

## Training STEP 1
1. Triplet DataLoader
2. Research: How can we finetune GPT-2 with custom made DataLoader?
3. Research: How can we parameter the size of generated paragraphs
4. Fine-tune GPT-2 Medium on a small dataset

## Evaluation STEP 1
1. Evaluate loss evolution during training
2. Benchmark (loss + BERT distance) against raw GPT-2 ; against GPT-2 with P1 only ; against GPT-2 with P1 and P3 only
3. Control of generated length
4. Control of theme consistency
5. Control of named entity occurence
6. Control of BERT consistency over P1-P2-P3

## Iterate over results
(PPLM, CTRL codes, graph based generation...)

## Webservice & Integration
1. Frontend
2. Backend
3. **DONE** - Fullstack integration
4. Algorithm integration + metrics
5. Distribution on server
6. Tests
7. Communication

## Final evaluation

## User evaluation

## Edit paper

## Publish & Distribute


