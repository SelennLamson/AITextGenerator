# Import libraries
from src.utils import *
import json
import os
from tqdm import tqdm
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts, list_supported_metadatas, get_metadata
from sortedcontainers import SortedDict
import zlib  # error message
from gutenberg._domain_model.exceptions import UnknownDownloadUriException  # error message


class DataPrepro:
    """
    Creates a json file for each book containing the text of book as well as its related metadata, including the genre,
    which we define ourselves from theme information
    Stores new json file in 'data/metadata/files'
    """

    def __init__(self):
        self.old_filename = 'clean_data.json'
        self.stats = dict()
        # Define literature genres - assign them keywords to be able to spot the genre from the theme of the book
        self.genres = SortedDict({
            'fiction': ['fiction', 'fictions'],
            'adventure': ['action', 'adventure', 'adventures'],
            'biography/history': ['biography', 'autobiography', 'history', 'historical', 'historical-fiction'],
            'children': ['child', 'tale', 'tales', 'children', 'baby', 'fairy',
                         'teenager', 'teen', 'teenage', 'young', 'juvenile'],
            'fantasy': ['fantasy', 'fantastic'],
            'romance': ['romance', 'love'],
            'science-fiction': ['sci-fy', 'science-fiction', 'scify', 'science'],
            'thriller': ['suspense', 'thriller', 'thrillers', 'horror', 'paranormal', 'ghost', 'mystery', 'mysteries',
                         'detective', 'crime', 'crimes', 'detectives', 'murder', 'police']
        })
        # Import json dataset
        try:
            self.data = json.load(open(METADATA_ROOT + self.old_filename, 'r'))
        except UnicodeDecodeError:
            self.data = json.load(open(METADATA_ROOT + self.old_filename, 'r', encoding='utf-8'))

    @staticmethod
    def find_real_genre(l, genres, new_el):
        """
        :param l: list of sentences indicating the theme of the book, gathered form gutenberg
        :param genres: dictionnary with keys: genres, keys: keywords to detect genre
        :param new_el: book considered (dictionnary)
        Find the genre(s) of a book from gutenberg theme information
        """
        #  Process list of themes - flatten it to see all keywords
        flat_list = list(set([item.lower() for sublist in l for item in sublist.split()]))

        # If keywords in 'theme' correspond to a genre, store it
        genre = []
        for el in flat_list:
            for i, sublist in enumerate(list(genres.values())):
                if el in sublist:
                    genre.append(genres.keys()[i])
                    genre = list(set(genre))
        # Create a new key with the genre(s) of the book
        new_el['genre'] = genre
        return new_el

    def create_json(self, b_id):
        """
        :param b_id: id of the book
        Create json file for each book (b_id)
        Don't for books where metadata is not accessible, cannot strip headers, is not in english, does not have a genre
        """
        genres = self.genres

        # Create new element that will be our new json file for the selected book -
        # choose this way to avoid computationally expensive storage
        # It will contain all information (text, id, author, title, genre, theme) and strips useless info at beginning
        new_el = self.data[str(b_id)]
        if 'en' in new_el['language']:  # keep only english files
            new_el['id'] = str(b_id)  # keep its id
            new_el = self.find_real_genre(new_el['theme'], genres, new_el)  # find genre
            self.data[str(b_id)]['genre'] = new_el['genre']  # add genre to original document
            try:
                new_el['text'] = strip_headers(load_etext(b_id)).strip()
                if len(new_el['genre']) != 0:  # do not include these files (not relevant for our study)
                    # Save it as a new json file if it is a novel
                    new_filename = str(b_id)
                    json.dump(new_el, open(METADATA_PATH + new_filename + METADATA_SUFFIX, 'w', encoding='utf-8'),
                              ensure_ascii=False, indent=1)
            except (
                    zlib.error,
                    UnknownDownloadUriException):  # deal with errors when importing text or removing headers
                # print('exception')
                pass

    @staticmethod
    def leave_one_genre(folder_data='data/preproc/'):
        books = os.listdir(folder_data)

        for book in tqdm(books):
            try:
                data = json.load(open(folder_data + book, 'r+'))
            except UnicodeDecodeError:
                print('WRONG ID, WRONG ID', book)
                pass

            # Change names
            mapping_names = {'biography': 'biography/history',
                             'history': 'biography/history',
                             'detective': 'thriller',
                             'mystery': 'thriller',
                             'horror': 'thriller',
                             'teen': 'children'}
            for i, genre in enumerate(data['genre']):
                if genre in list(mapping_names.keys()):
                    data['genre'][i] = mapping_names[genre]
                if genre in ['short stories', 'english', 'satire', 'western']:
                    data['genre'].pop(i)

            # Preprocess genre again to leave only one
            if len(data['genre']) > 1:
                for genre in ['fiction', 'children', 'adventure', 'biography/history', 'romance', 'thriller',
                              'science-fiction']:
                    if genre in data['genre'] and len(data['genre']) != 1: data['genre'].remove(genre)
            if len(data['genre']) == 0:
                data['genre'] = ['fiction']

            # Reprocess correctly science fiction
            if data['genre'][0] == 'fiction':
                flat_list = list(set([item.lower() for sublist in data['theme'] for item in sublist.split()]))
                if 'science' in flat_list:
                    data['genre'] = ['science-fiction']

            json.dump(data, open(folder_data + book, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)

    def stats_genre(self, folder_data='data/preproc/'):
        """
        :param folder_data: folder where data is placed
        :return: dictionary giving information on genre repartition
        """
        books = os.listdir(folder_data)

        for genre in list(self.genres.keys()):
            self.stats[genre] = 0

        for book in tqdm(books):
            try:
                data = json.load(open(folder_data + book, 'r+'))
            except UnicodeDecodeError:
                print('WRONG ID, WRONG ID', book)
                pass

            self.stats[data['genre'][0]] += 1

        return self.stats
