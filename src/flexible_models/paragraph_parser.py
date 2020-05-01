from typing import Dict
import re
from collections import OrderedDict
import matplotlib.pyplot as plt

from .flexible_model import FlexibleModel
from src.utils import *


class ParagraphParser(FlexibleModel):
    def __init__(self, min_threshold=20, min_length=600, max_length=900):
        """
        Initializes a paragraph parser.
        :param min_threshold: minimum length (in chars) a paragraph should be to be taken into account.
                          Lower than this threshold often means it's a title or a chapter separator.
        :param min_length: minimum length of final sub-paragraphs. It will not be strictly respected though.
        :param max_length: maximum length of final sub-paragraphs. Strictly respected.
        """
        super().__init__()
        self.min_threshold = min_threshold
        # self.min_length = min_length
        # self.max_length = max_length

        self.size_index = 0
        self.prev_size_index = 0

    def rand_size(self):
        self.prev_size_index = self.size_index
        self.size_index = random.randint(0, len(SIZES) - 1)

    @property
    def prev_min_length(self):
        return (SIZES[self.prev_size_index].inf_chars + SIZES[self.prev_size_index].sup_chars) // 2

    @property
    def prev_max_length(self):
        return SIZES[self.prev_size_index].sup_chars

    @property
    def min_length(self):
        return (SIZES[self.size_index].inf_chars + SIZES[self.size_index].sup_chars) // 2

    @property
    def max_length(self):
        return SIZES[self.size_index].sup_chars

    def predict(self, full_text: str, ents_p: Dict[str, str], ents_o: Dict[str, str], ents_l: Dict[str, str],
                ents_m: Dict[str, str], verbose: int = 0) -> List[Dict[str, Any]]:
        """
        Splits a text into paragraphs.
        :param full_text: string to summarize.
        :param ents_p: dictionnary of {position: entity} persons
        :param ents_o: dictionnary of {position: entity} organisations
        :param ents_l: dictionnary of {position: entity} locations
        :param ents_m: dictionnary of {position: entity} miscellaneous
        :param verbose: 0 for silent execution, 1 for statistics and 2 for statistics and histogram of paragraphs sizes.
        :return: list of paragraphs
        """

        all_p = set(ents_p.values()).difference({'M', 'Mr', 'Ms', 'The', 'Dr'})
        all_o = set(ents_o.values()).difference({'M', 'Mr', 'Ms', 'The', 'Dr'})
        all_l = set(ents_l.values()).difference({'M', 'Mr', 'Ms', 'The', 'Dr'})
        all_m = set(ents_m.values()).difference({'M', 'Mr', 'Ms', 'The', 'Dr'})

        classes = dict()
        classes['persons'] = OrderedDict(
            {key: elt for key, elt in sorted([(int(key), elt) for key, elt in ents_p.items()], key=lambda x: x[0])})
        classes['organisations'] = OrderedDict(
            {key: elt for key, elt in sorted([(int(key), elt) for key, elt in ents_o.items()], key=lambda x: x[0])})
        classes['locations'] = OrderedDict(
            {key: elt for key, elt in sorted([(int(key), elt) for key, elt in ents_l.items()], key=lambda x: x[0])})
        classes['misc'] = OrderedDict(
            {key: elt for key, elt in sorted([(int(key), elt) for key, elt in ents_m.items()], key=lambda x: x[0])})

        paragraphs = []

        # Display some information about the novel
        if verbose >= 1:
            print("Text:\t\"", full_text[:100].replace('\n', ' ') + "...\"")

        # Parsing paragraphs
        def add_paragraph(content):
            c = content.strip()

            p_data = {
                         'size': len(c),
                         'text': c,
                         'summaries': list()
                     } + {ent_class: [] for ent_class in ENTITY_CLASSES}

            search_content = content.replace(',', ' ').replace('"', '').replace("'", '').replace(';', '').replace('_',
                                                                                                                  '').replace(
                '”', '').replace('“', '')

            p_data['persons'] = [p for p in all_p if p in search_content]
            p_data['organisations'] = [o for o in all_o if o in search_content]
            p_data['locations'] = [l for l in all_l if l in search_content]
            p_data['misc'] = [m for m in all_m if m in search_content]

            paragraphs.append(p_data)

        # Removing any isolated line-breaks, any multiple whitespaces and separating text into real paragraphs
        striped_of_linebreaks = ' '.join('\n' if elt == '' else elt for elt in full_text.split('\n'))
        striped_of_multispaces = re.sub(r'[ ]+', ' ', striped_of_linebreaks)
        real_paragraphs = [elt.strip() for elt in striped_of_multispaces.split('\n') if elt != '']

        # Splitting the book in parts (= sequence of consecutive paragraphs)
        # Every paragraph that is less than min_threshold length is considered a separator:
        # It is discarded and marks the end of current part and the beginning of the next one.
        parts = []
        current_part = []
        for i, p in enumerate(real_paragraphs):
            if len(p) <= self.min_threshold:
                if len(current_part) > 0:
                    parts.append(current_part)
                    current_part = []
                continue
            current_part.append(p)
        if len(current_part) > 0:
            parts.append(current_part)

        # Computing statistics on book
        content_length = 0
        if verbose >= 1:
            avg_paragraphs_per_part = sum(len(part) for part in parts) / len(parts)
            content_length = sum(sum(len(paragraph) for paragraph in part) for part in parts)
            num_paragraphs = sum(len(part) for part in parts)
            avg_paragraph_length = content_length / num_paragraphs
            print("\n--- NOVEL STATISTICS ---")
            print("Content length:\t\t\t\t", content_length)
            print("Parts:\t\t\t\t\t\t", len(parts))
            print("Paragraphs:\t\t\t\t\t", num_paragraphs)
            print("Paragraphs per part:\t\t {:.2f}".format(avg_paragraphs_per_part))
            print("Average paragraph length:\t {:.2f}".format(avg_paragraph_length))

        # Concatenating small parts (< MIN_LENGTH) with the next one, because it probably comes from wrong parsing
        i = 0
        while i < len(parts):
            if sum(len(p) for p in parts[i]) >= self.min_length:
                i += 1
            elif i < len(parts) - 1:
                part = parts.pop(i)
                parts[i] = part + parts[i]
            else:
                break

        # Beginning the splitting of paragraphs
        for part_i, part in enumerate(parts):
            for pi, rp in enumerate(part):
                if rp == "":
                    continue

                # Lookahead: to prevent small trail paragraphs, we look ahead and add to current paragraph
                # the next ones as long as they are less than the minimum length.
                while pi + 1 < len(part) and len(part[pi + 1]) < self.min_length:
                    np = part.pop(pi + 1)
                    rp = rp + ' ' + np

                # Paragraph is of correct size, or is too short but is the last one of current part --> Add it as is.
                if self.min_length <= len(rp) <= self.max_length or (len(rp) < self.min_length and pi == len(part) - 1):
                    add_paragraph(rp)

                # Paragraph is too short --> Add it to the beginning of the next one.
                elif len(rp) < self.min_length:
                    part[pi + 1] = rp + part[pi + 1]

                # Paragraph is too long --> Split it in parts and add the remaining to the next one.
                else:
                    # Splitting the paragraph at target_length, without cutting words
                    #  1. We add words until we pass MIN_LENGTH
                    #  2. If we encounter a final point, we end the split. --> NEXT (with empty)
                    #  3. When we pass target_length, we keep a memory of the current paragraph.
                    #  4. If we encounter a final point, we end the split. --> NEXT (with empty)
                    #  5. When we pass MAX_LENGTH, we end the split with step 3's memory. --> NEXT (with current - saved)
                    #  6. If end of paragraph and we are below MIN_LENGTH --> SPECIAL CONDITIONS
                    to_add = []
                    current_split = ""
                    saved_split = ""
                    words = rp.split(' ')
                    for wi, w in enumerate(words):
                        current_split = (current_split + " " + w).strip()
                        final_point = current_split[-1] in ['.', '!', '?', '\'', '"']
                        if self.min_length <= len(current_split) <= self.max_length and final_point:
                            to_add.append(current_split)
                            self.rand_size()
                            current_split = ""
                            saved_split = ""
                        elif self.max_length <= len(current_split):
                            to_add.append(saved_split.strip())
                            self.rand_size()
                            current_split = current_split[len(saved_split):].strip()
                            saved_split = ""
                        elif (self.min_length + self.max_length) / 2 <= len(current_split) and saved_split == "":
                            saved_split = current_split
                        elif wi == len(words) - 1:  # current_split is too small, and it's the end of the paragraph
                            # We check if we can put it back to the previous paragraph
                            if len(to_add) > 0 and len(to_add[-1]) + len(current_split) + 1 <= self.prev_max_length:
                                to_add[-1] += ' ' + current_split
                            # Otherwise, we check if there is a paragraph after the current one
                            elif pi < len(part) - 1:
                                part[pi + 1] = current_split + ' ' + part[pi + 1]
                            # Finally, we cannot add it back to previous or to next one, so we add it as is
                            else:
                                to_add.append(current_split.strip())
                                self.rand_size()
                    for p in to_add:
                        add_paragraph(p)

        # Printing results
        if verbose >= 1:
            sizes = []
            for p in paragraphs:
                sizes.append(p['size'])

            print("\n--- EXTRACTED GROUPS ---")
            print('Groups:\t\t\t\t', len(sizes))
            print('Average length:\t\t', int(sum(sizes) / len(sizes)))
            print('Max length:\t\t\t', max(sizes))
            print('Min length:\t\t\t', min(sizes))
            print('% of raw text:\t\t {}%'.format(int(100 * sum(sizes) / len(full_text))))
            print('% of clean text:\t {}%'.format(int(100 * sum(sizes) / content_length)))
            print('\n-- First paragraph:\n"' + paragraphs[0]['text'][:100] + '..."')

            if verbose >= 2:
                plt.hist(sizes, bins=50)
                plt.xlabel("Paragraph sizes")
                plt.ylabel("# paragraphs")
                plt.show()

        return paragraphs
