import time
from src.utils import *
from src.flexible_models.flexible_bert_ner import FlexibleBERTNER


def perform_global_ner_on_all(model: FlexibleBERTNER, files: List[str] = None, verbose: int = 1):
    """
    Applies NER model on all metadata/file/id.json files, replacing already existing entities in the way.
    :param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
    :param files: optional list of files to work on
    :param verbose: 0 for silent execution, 1 to display progress.
    """
    files = os.listdir(METADATA_PATH) if files is None else [f + METADATA_SUFFIX for f in files]
    for f in files:
        d_id = f[:-len(METADATA_SUFFIX)]
        if not os.path.exists(METADATA_PATH + d_id + METADATA_SUFFIX):
            continue
        if verbose >= 1:
            print("Processing file:", f)

        now = time.time()
        perform_global_ner_on_file(model, d_id, verbose)
        print("Time elapsed: {}s".format(int(time.time() - now)))


def perform_global_ner_on_file(model: FlexibleBERTNER, d_id: str = None, verbose: int = 1):
    """
    Applies NER model on all a metadata/file/id.json file, replacing already existing entities in the way.
    :param model: the FlexibleBERTNER model to apply, should have a predict(text) method.
    :param d_id: file id.
    :param verbose: 0 for silent execution, 1 to display progress.
    """

    # Input of file ID
    if d_id is None:
        while True:
            d_id = input("Select a novel id: ")
            if os.path.exists(METADATA_PATH + d_id + METADATA_SUFFIX):
                break
            print("ERROR - Id", d_id, "not found.")

    # Reading JSON file
    novel_data = json.load(open(METADATA_PATH + d_id + METADATA_SUFFIX, 'r', encoding='utf-8'))
    text = novel_data['text'].replace('\n', ' ')

    output = model.predict_with_index(text, verbose)

    persons = dict()
    locations = dict()
    organisations = dict()
    misc = dict()
    for pi, (index, entity, tag) in enumerate(output):
        if verbose >= 1:
            print("\rNER outputs - {:.2f}%".format(pi / len(output) * 100), end="")

        if tag == "PER":
            persons[index] = entity
        elif tag == "LOC":
            locations[index] = entity
        elif tag == "ORG":
            organisations[index] = entity
        elif tag == "MISC":
            misc[index] = entity

    novel_data['persons'] = persons
    novel_data['locations'] = locations
    novel_data['organisations'] = organisations
    novel_data['misc'] = misc

    # Saving JSON file
    json.dump(novel_data, open(NOVEL_PATH + d_id + NOVEL_SUFFIX, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    if verbose >= 1:
        print("\rNER outputs - 100%")
