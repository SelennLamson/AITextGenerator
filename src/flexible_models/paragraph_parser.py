from typing import List, Any, Dict
import re
import matplotlib.pyplot as plt

from .flexible_model import FlexibleModel


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
		self.min_length = min_length
		self.max_length = max_length

	def predict(self, full_text:str, verbose:int = 0) -> List[Dict[str, Any]]:
		"""
		Performs NER on strings of any length.
		:param full_text: string to summarize.
		:param verbose: 0 for silent execution, 1 for statistics and 2 for statistics and histogram of paragraphs sizes.
		:return: list of paragraphs
		"""
		target_length = (self.min_length + self.max_length) / 2
		paragraphs = []

		# Display some information about the novel
		if verbose >= 1:
			print("Text:\t\"", full_text[:100].replace('\n', ' ') + "...\"")

		# Parsing paragraphs
		def add_paragraph(content):
			c = content.strip()
			assert len(c) <= self.max_length
			paragraphs.append({
				'size': len(c),
				'text': c
			})

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
							current_split = ""
							saved_split = ""
						elif self.max_length <= len(current_split):
							to_add.append(saved_split.strip())
							current_split = current_split[len(saved_split):].strip()
							saved_split = ""
						elif target_length <= len(current_split) and saved_split == "":
							saved_split = current_split
						elif wi == len(words) - 1:  # current_split is too small, and it's the end of the paragraph
							# We check if we can put it back to the previous paragraph
							if len(to_add) > 0 and len(to_add[-1]) + len(current_split) + 1 <= self.max_length:
								to_add[-1] += ' ' + current_split
							# Otherwise, we check if there is a paragraph after the current one
							elif pi < len(part) - 1:
								part[pi + 1] = current_split + ' ' + part[pi + 1]
							# Finally, we cannot add it back to previous or to next one, so we add it as is
							else:
								to_add.append(current_split.strip())
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
				plt.hist(sizes)
				plt.xlabel("Paragraph sizes")
				plt.ylabel("# paragraphs")
				plt.show()

		return paragraphs

