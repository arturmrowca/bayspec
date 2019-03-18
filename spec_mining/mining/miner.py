from abc import ABCMeta, abstractmethod
from spec_mining.specification.regex_specification import SymbolType
from time import strftime, localtime


class Miner(object, metaclass=ABCMeta):

    @abstractmethod
    def start(self):
        """
        :return:
        """

    @staticmethod
    def get_time_string():
        return strftime("%Y_%m_%d-%H_%M_%S", localtime())

    @staticmethod
    def _get_spec_metrics(spec):
        defaults = {"length": 0, "abs_matches": 0,
                    "rel_matches": 0, "n_matching_strings": 0}

        if len(spec) == 0:
            return defaults

        non_options = [i for i, s in enumerate(spec) if s.type != SymbolType.Option]

        if not non_options:
            return defaults

        first_index = min(non_options)
        last_index = max(non_options)

        if first_index >= last_index:
            return defaults

        _spec = spec[first_index: last_index + 1]

        # Metrics
        length = len(_spec)
        abs_matches = len([0 for s in _spec if s.type == SymbolType.Letter])
        rel_matches = abs_matches / length

        n_matching_strings = 1
        for symbol in _spec:
            if symbol.type == SymbolType.Group:
                n_matching_strings *= len(symbol.symbol)
            elif symbol.type == SymbolType.Option:
                n_matching_strings *= (len(symbol.symbol) + 1)

        return {"length": length,
                "abs_matches": abs_matches,
                "rel_matches": rel_matches,
                "n_matching_strings": n_matching_strings,
                }
