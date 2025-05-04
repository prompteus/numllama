from __future__ import annotations

import math
import random
import re
from typing import Optional

import bs4
from bs4 import Tag
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import numllama as gadgets  # The only change compared to steps_utils.py in git:calc-x/steps-consistency


def separate_chain_to_steps(chain: str, special_sep_token: Optional[str] = None) -> tuple[list[str], str]:
    """
    heuristically separates input chain into a list of reasoning steps.
    :param chain: Original chain
    :param special_sep_token: Additional, model-specific separator token
    :return: A tuple: (list of steps contained in the chain, used separator)
    """
    # pick the first candidate sep token that occurs in the chain
    sep = special_sep_token if (special_sep_token is not None and special_sep_token in chain) \
        else ". " if ". " in chain else ".\n" if ".\n" in chain else "\n\n" if "\n\n" in chain else "\n"

    steps = chain.split(sep)

    return steps, sep


class StepPermuter:
    numeral_re = re.compile(r"\d+(?:\.\d+)?")

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def _replace_num(self, number: int | float, contains_exp: bool = False) -> str:
        # replace with a number of a similar scale as the original
        is_decimal = "." in str(number)
        number_length = len(str(number).replace("_", "").split(".")[0]) if is_decimal else len(str(number))

        output_number = random.randint(1, 10 ** (number_length + (-1 if contains_exp else 1)))
        if is_decimal:
            output_number += (random.randrange(100) / 100)

        return str(output_number)

    @staticmethod
    def _replace_all(text: str, replacement_map: dict[str, str]) -> str:
        out_text = text
        for orig, repl in replacement_map.items():
            out_text = out_text.replace(orig, repl)
        return out_text

    def _permute_numbers_all_steps(self,
                                   sample_steps: list[str],
                                   supported_range_start: int = 0,
                                   supported_range_end: int = 130_000) -> tuple[list[str], str]:
        calculator = gadgets.gadget.Calculator()
        question = sample_steps[0]
        # we assume that the given questions already do not contain options -- we have
        multi_choice_seps = ("Choose the correct option:", "Pick one:")

        try:
            multi_choice_sep = next(sep for sep in multi_choice_seps if sep in question)
            question_without_choices = question.split(multi_choice_sep)[0]
        except StopIteration:
            # not a multi-choice question
            multi_choice_sep = None
            question_without_choices = question

        all_results_positive = False
        num_iters = 0
        while not all_results_positive:
            all_results_positive = True  # passing the non-repeat condition unless the check of new results fails
            # permute numbers in the question (first step)
            first_step_numerals = self.numeral_re.findall(question_without_choices)
            replaces_map = {num: self._replace_num(num, contains_exp=any("**" in step for step in sample_steps))
                            for num in first_step_numerals}

            out_steps = [self._replace_all(question, replaces_map)]

            last_result = None
            # for the reasoning steps, replace the inputs according to the input question
            # + <outputs> of previous steps computed from the already-altered inputs
            # both replacements are performed after we recompute the new_gadget_output
            for step in sample_steps[1:]:
                step_altered = self._replace_all(step, replaces_map)

                doc = bs4.BeautifulSoup(step_altered, features="html.parser")
                doc_orig = bs4.BeautifulSoup(step_altered, features="html.parser")
                gadget_tags: list[bs4.Tag] = doc.find_all(gadgets.markup.GADGET_TAG)
                output_tags: list[bs4.Tag] = doc_orig.find_all(gadgets.markup.OUTPUT_TAG)

                for gadget_tag_input, orig_output in zip(gadget_tags, output_tags):
                    # next_el = gadget_tag_input.next_sibling
                    # next_out = orig_output.next_sibling
                    # # skip whitespaces before the call
                    # while next_el is not None and isinstance(next_el, bs4.NavigableString) and next_el.get_text().strip() == "":
                    #     next_el = next_el.next_sibling
                    # while next_out is not None and isinstance(next_out, bs4.NavigableString) and next_out.get_text().strip() == "":
                    #     next_out = orig_output.next_sibling
                    gadget_id = gadget_tag_input.get("id", None)
                    if gadget_id != calculator.gadget_id():
                        # we extract permuted numerals only from Calculator computations
                        # TODO: because of this, multi-choice answers are not registered and replaced in output
                        continue

                    gadget_input = gadget_tag_input.get_text()
                    orig_gadget_output = orig_output.get_text()
                    # if "**" in gadget_input and gadget_input.count("_") > 3:
                    #     print("Calc input:" + gadget_input)
                    #     continue
                    if "**" in gadget_input:
                        try:
                            base, exp = (float(i) for i in gadget_input.split("**"))
                            if base >= math.sqrt(supported_range_end) or exp > 10:
                                # this would cause the calculator to halt, so we skip the whole permutation alltogether
                                all_results_positive = False
                                break
                        except ValueError:
                            # ValueError: could not convert string to float: ' (1/2)'
                            # -> usually caused by fraction exponents, not causing calculator overflow
                            continue
                    elif "factorial" in gadget_input:
                        # in the case of factorial, this will most likely explode
                        all_results_positive = False
                        break
                    try:
                        new_gadget_output = calculator(gadget_input, add_approx=False)
                    except KeyboardInterrupt:
                        print("Interrupted on input %s" % gadget_input)
                        raise
                    last_result = new_gadget_output
                    try:
                        new_output_float = float(new_gadget_output)
                        if supported_range_start <= new_output_float <= supported_range_end:
                            all_results_positive = False
                    except ValueError:
                        # output unparseable as number -> check positivity only as string
                        if new_gadget_output.startswith("-"):
                            all_results_positive = False

                    replaces_map[orig_gadget_output] = new_gadget_output.split(" = around")[0]

                out_steps.append(self._replace_all(step, replaces_map))

            if not all_results_positive:
                num_iters += 1
            if num_iters > 5:
                # print("Skipping altering a chain because of more than 5 unsuccessful attempts.")
                break

        # print("Constructed altered chain in %s attempts." % num_iters)
        # if multi_choice_sep is not None:
        #     # replace the original options with the occurrence of the correct result on the same position
        #     # question, options = out_steps[0].split("Pick one:", maxsplit=2)
        #     # out_steps[0] = "Pick one:".join([question, self._replace_all(options, replaces_map)])
        #
        #     out_steps[0] = question_without_choices

        return out_steps, last_result  # altered steps + the most-recent, altered output (=final result)

    choice_map = ["A", "B", "C", "D", "E", "$"]

    def _replace_choice(self, step: str, choice: str, value: str) -> str:
        # replaces the value of choice in the given step with the passed value
        end_choice = self.choice_map[self.choice_map.index(choice)+1]
        orig_result = re.findall("%s[^a-zA-Z0-9]+(.+?)[^a-zA-Z0-9]*%s" % (choice, end_choice), step)[0]
        return step.replace(orig_result, value)

    def permute_all_steps(self, sample_steps: list[str]) -> list[str]:
        # step_str = self.tokenizer.batch_decode(input_ids)
        output_steps, new_result = self._permute_numbers_all_steps(sample_steps)
        if new_result is not None:
            # any steps were updated
            assert "<result>" in sample_steps[-1], "The last step in the sequence must contain <result> tag."

            new_result_elem = bs4.BeautifulSoup(sample_steps[-1], features="html.parser")
            result_value_or_choice = next(str(el.contents[0]) for el in new_result_elem.contents
                                          if isinstance(el, bs4.Tag) and el.name == "result")

            if result_value_or_choice.strip().upper() in self.choice_map:
                # multiple-choice -> replace the value at the corresponding choice with the new result
                output_steps[0] = self._replace_choice(output_steps[0], result_value_or_choice, new_result)

        return output_steps

    def permute_chain(self, question: str, chain: str) -> tuple[str, str]:
        separated_steps, sep = separate_chain_to_steps(chain)
        out_steps = self.permute_all_steps([question, *separated_steps])
        return out_steps[0], sep.join(out_steps[1:])
