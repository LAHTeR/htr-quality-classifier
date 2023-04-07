from typing import List
from .scorer import Scorer


class GarbageDetector(Scorer):
    _VOWELS = "aäàáâǎeéèêëěiîïíìıoöôòóǒuüûùúǔ"

    EPR_RULE1 = 21
    EPR_RULE2 = 3
    EPR_RULE3 = 4
    EPR_RULE4 = 6
    EPR_RULE5 = 8
    EPR_RULE9 = 2

    def score(self, tokens: List[str]) -> float:  # noqa: MC0001
        """
        `See Nautilus-OCR <https://github.com/natliblux/nautilusocr/blob/2d4d59c45466b5cc8c9897798bd8b205a7f0c02c/src/epr/features_epr.py#L148>`_
        """
        # pylint: disable=consider-using-enumerate,too-many-branches,too-many-locals,too-many-statements,chained-comparison

        issues = 0

        if len(tokens) == 0:
            return 0

        for token in tokens:

            # rule1
            if len(token) >= GarbageDetector.EPR_RULE1:
                issues += 1
                continue

            vowel_count = 0
            consonant_count = 0
            lower_case_count = 0
            upper_case_count = 0
            special_char_count = 0
            non_outer_special_chars = set()
            alpha = True
            last_char = None
            repitition_streak = 0
            vowel_streak = 0
            consonant_streak = 0
            go_to_next_token = False
            for i in range(0, len(token)):
                go_to_next_token = False
                char = token[i]

                # collect token info
                if char.isalpha():
                    if char.lower() in GarbageDetector._VOWELS:
                        vowel_count += 1
                        vowel_streak += 1
                        consonant_streak = 0
                    else:
                        consonant_count += 1
                        consonant_streak += 1
                        vowel_streak = 0
                    if char.isupper():
                        upper_case_count += 1
                    else:
                        lower_case_count += 1
                elif char.isalnum():
                    alpha = False
                    vowel_streak = 0
                    consonant_streak = 0
                else:
                    special_char_count += 1
                    alpha = False
                    vowel_streak = 0
                    consonant_streak = 0
                    # pylint: disable=consider-using-in
                    if i != 0 and i != len(token) - 1:
                        non_outer_special_chars.add(char)

                # rule 3
                if vowel_streak >= GarbageDetector.EPR_RULE3:
                    issues += 1
                    go_to_next_token = True
                    break

                # rule 4
                if consonant_streak >= GarbageDetector.EPR_RULE4:
                    issues += 1
                    go_to_next_token = True
                    break

                if last_char is not None and char == last_char:
                    repitition_streak += 1

                    # rule 2
                    if repitition_streak >= GarbageDetector.EPR_RULE2:
                        issues += 1
                        go_to_next_token = True
                        break
                else:
                    repitition_streak = 0
                last_char = char

            if go_to_next_token:
                continue

            if alpha and vowel_count > 0 and consonant_count > 0:
                # rule 5
                if vowel_count * GarbageDetector.EPR_RULE5 < consonant_count:
                    issues += 1
                    continue
                # rule 5
                if consonant_count * GarbageDetector.EPR_RULE5 < vowel_count:
                    issues += 1
                    continue

            # rule 6
            if lower_case_count > 0 and upper_case_count > lower_case_count:
                issues += 1
                continue

            # rule 7
            if (
                upper_case_count > 0
                and token[0].islower()
                and token[len(token) - 1].islower()
            ):
                issues += 1
                continue

            # rule 8
            regular_chars = len(token) - special_char_count
            if special_char_count >= regular_chars and regular_chars > 0:
                issues += 1
                continue

            # rule 9
            if len(non_outer_special_chars) >= GarbageDetector.EPR_RULE9:
                issues += 1
                continue

        return issues / len(tokens)
