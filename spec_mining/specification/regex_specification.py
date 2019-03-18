from enum import Enum
from spec_mining.utility.levenshtein import Levenshtein, EditOperation
from spec_mining.specification.abstract_specification import Specification
from pyModelChecking.CTLS import And, Or, Imply
from pyModelChecking.CTL import EU, EX
from pyModelChecking.LTL import F, U, X, G, AtomicProposition

import numpy as np


class RegexSpec(Specification, list):

    def __init__(self, path):
        super().__init__()
        for node in path:
            self.append(Symbol(node, SymbolType.Letter))

    def merge(self, spec, Xable):
        matrix = self.levenshtein_matrix(spec)
        edit_operations = Levenshtein().levenshtein_backtrace(matrix)

        insert_indices = []
        path_index = 0
        prev_operation = EditOperation.Match
        for i, operation in enumerate(edit_operations):

            # DELETE
            if operation == EditOperation.Delete.value:
                self._delete(i - len(insert_indices))
                prev_operation = EditOperation.Delete

            # SUBSTITUTE
            elif operation == EditOperation.Substitute.value:
                self._substitute(i - len(insert_indices), spec[path_index])
                if prev_operation == EditOperation.Insert:
                    self[i - len(insert_indices)].Xable = Xable[path_index]
                elif prev_operation != EditOperation.Delete:
                    if not Xable[path_index]:
                        self[i - len(insert_indices)].Xable = False
                prev_operation = EditOperation.Substitute
                path_index += 1

            # INSERT
            elif operation == EditOperation.Insert.value:
                insert_indices.append((i, path_index))
                prev_operation = EditOperation.Insert
                path_index += 1

            # MATCH
            else:
                self[i - len(insert_indices)].count += 1
                self[i - len(insert_indices)].symbol[spec[path_index]] += 1
                if prev_operation == EditOperation.Insert:
                    self[i - len(insert_indices)].Xable = Xable[path_index]
                elif prev_operation != EditOperation.Delete:
                    # if previous symbol was deleted, we do not change Xable as we want to keep
                    # the Xable from the deleted symbol
                    if not Xable[path_index]:
                        self[i - len(insert_indices)].Xable = False
                path_index += 1
                prev_operation = EditOperation.Match

        # INSERT - Part 2
        for (insert_pos, path_index) in insert_indices:
            self.insert(insert_pos, Symbol(spec[path_index], SymbolType.Option, count=1))
            self[insert_pos].Xable = Xable[path_index]

    def to_CTL(self):
        if len([s for s in self if s.type != SymbolType.Option]) <= 1:
            return None

        # maximum index of a symbol that is not of type Option (?)
        last_index = max([i for i, s in enumerate(self) if s.type != SymbolType.Option])
        return self._rec_to_CTL(0, last_index)

    def _rec_to_CTL(self, index, last_index):
        if index >= last_index:
            return self._symbol_to_formula(index)

        if self[index].type == SymbolType.Option:
            return EU(self._symbol_to_formula(index), self._rec_to_CTL(index + 1, last_index))

        return And(self._symbol_to_formula(index), EX(self._rec_to_CTL(index + 1, last_index)))

    def _symbol_to_formula(self, index):
        key_list = [*self[index].symbol]
        if self[index].type == SymbolType.Letter:
            return key_list[0]

        key_list = [*self[index].symbol]
        if len(key_list) == 1:
            return key_list[0]

        phi = Or(key_list[0], key_list[1])
        for symbol in key_list[2:]:
            phi = Or(phi, symbol)
        return phi

    def _literals_to_formula(self, symbol):
        literal_list = [*symbol.symbol]
        if len(literal_list) == 1:
            return literal_list[0]

        phi = Or(literal_list[0], literal_list[1])
        for symbol in literal_list[2:]:
            phi = Or(phi, symbol)
        return phi

    def to_LTL(self):
        # maximum index of a symbol that is not of type Option (?)
        non_options = [i for i, s in enumerate(self) if s.type != SymbolType.Option]
        if not non_options:
            return None

        first_index = min(non_options)
        last_index = max(non_options)

        if first_index >= last_index:
            return None

        # get premise index
        try:
            max_premise_index = self._get_last_premise_index(first_index, last_index)
        except LTLException:
            return None

        if max_premise_index >= last_index:
            return None

        # L().log.info("max_premise_index = {}".format(max_premise_index))

        # compute conclusion
        conclusion = self._rec_to_LTL(max_premise_index + 1, last_index)

        # compute premise
        premise = conclusion
        for i,s in enumerate([s for s in self[:max_premise_index+1] if s.type != SymbolType.Option][::-1]):
            if i == 0:
                premise = Imply(self._literals_to_formula(s), premise)
            else:
                premise = Imply(self._literals_to_formula(s), X(G(premise)))

        formula = G(premise)

        return formula

    def _get_last_premise_index(self, first_index, last_index):
        def checker(i, j):
            if j > last_index:
                return i

            # search next start
            while self[i].type == SymbolType.Option:
                i += 1

            result2 = 0
            if self[j].type == SymbolType.Option:
                result2 = checker(i, j+1)

            while any(symb in self[j].symbol for symb in self[i].symbol):
                j += 1
                i += 1
                if i >= last_index:
                    raise LTLException()
                while self[i].type == SymbolType.Option:
                    i += 1
                if i >= last_index:
                    raise LTLException()
                if j > last_index:
                    break

            result1 = i
            return max(result1, result2)
        # ---- end of function definition ----

        max_premise = 0
        for k in range(first_index + 1, last_index + 1):
            if any(symb in self[k].symbol for symb in self[0].symbol):
                result = checker(1, k+1)
                if max_premise < result:
                    max_premise = result

        return max_premise

    def _rec_to_LTL(self, index, last_index):
        """
        :param index:
        :param last_index:
        :return:
        """
        if index > last_index:
            return None

        if index == last_index:
            phi = self._symbol_to_formula(index)
            if self[index].Xable:
                return X(phi)
            else:
                return X(F(phi))

        if self[index].type == SymbolType.Option:
            j = index
            start_index = index
            first_is_X = self[index].Xable
            while self[j].type == SymbolType.Option:
                if not self[j + 1].Xable:
                    first_is_X = False
                    start_index = j + 1
                j += 1

            if j == start_index:
                # NO Until
                self[j].Xable = False
                return self._rec_to_LTL(j, last_index)
            else:
                # Until starting from start_index
                psi = self._rec_to_LTL(j + 1, last_index)
                if psi:
                    # o1 U o2 U ... U (pi And psi)
                    phi = And(self._symbol_to_formula(j), psi)
                else:
                    # o1 U o2 U ... U phi
                    phi = self._symbol_to_formula(j)

                for i in list(range(start_index, j))[::-1]:
                    phi = U(self._symbol_to_formula(i), phi)

                if first_is_X:
                    return X(phi)
                else:
                    return X(F(phi))

        phi = And(self._symbol_to_formula(index), self._rec_to_LTL(index + 1, last_index))
        if self[index].Xable:
            return X(phi)
        else:
            return X(F(phi))


    def levenshtein_matrix(self, path):
        height = len(self) + 1
        width = len(path) + 1
        matrix = np.ndarray(shape = (height,width))
        for i in range(width):
            matrix[0,i] = i
        for i in range(1, height):
            matrix[i,0] = i

        for i in range(1, height):
            for j in range(1, width):
                sub_cost = 0 if self._check_letter_in_regex(i-1, path[j-1]) else Levenshtein().SUBSTITUTION_COST
                matrix[i, j] = min(matrix[i - 1, j    ] + Levenshtein().DELETION_COST,  # deletion
                                   matrix[i    , j - 1] + Levenshtein().INSERTION_COST,  # insertion
                                   matrix[i - 1, j - 1] + sub_cost)             # substitution
        return matrix

    def _check_letter_in_regex(self, index, letter):
        return letter in self[index].symbol

    def _delete(self, index):
        self[index].count += 1
        if self[index].type != SymbolType.Option:
            self[index].type = SymbolType.Option

    def _substitute(self, index, new_symbol):
        self[index].symbol[new_symbol] = 1
        self[index].count += 1
        if self[index].type == SymbolType.Letter:
            # for Letter, transform to Group and add new symbol
            self[index].type = SymbolType.Group

    def to_string(self):
        non_options = [i for i, s in enumerate(self) if s.type != SymbolType.Option]

        if not non_options:
            return "<empty spec>"

        first_index = min(non_options)
        last_index = max(non_options)
        string = ""
        for symbol in self[first_index: last_index + 1]:
            if symbol.Xable:
                string += "<X>"
            else:
                string += "<F>"

            if symbol.type == SymbolType.Letter:
                string += "(" + "".join(symbol.symbol) + ")"
            elif symbol.type == SymbolType.Option:
                string += "(" + "|".join(symbol.symbol) + ")?"
                # string = "{0}({1})?".format(string, "|".join(symbol.symbol))
            else:
                string += "(" + "|".join(symbol.symbol) + ")"
                # string = "{0}({1})".format(string, "|".join(symbol.symbol))
        return string

    @staticmethod
    def unittest_to_temporal_logic():
        testsuite = [["ABCDEF", "AbCDeF", "aBCDEf"],
                     ["ACE", "BDE", "CE"],
                     ["ABD", "aBD", "Abd", "AD"],
                     ["ABD", "AD", "aBD", "Abd"],
                     ["abx", "acy", "ab"],
                     ["abdf", "acef", "abf", "adf"],
                     ["acxy", "bxy", "cxy"]]

        for i, test in enumerate(testsuite):
            print("Test #{0}:".format(i + 1))
            for string in test:
                re = RegexSpec(string)
                print("{0}:".format(re.to_string()))
                # print("CTL: {0}".format(re.to_CTL()))
                # print("LTL: {0}".format(re.to_LTL()))
            print("")

    @staticmethod
    def evaluation_metrics(spec):
        height = spec.height - 1

        leaves = []
        RegexSpec._spec_leaves(spec, leaves)

        width = len(leaves)
        unique = len(set(leaves))

        return {"height": height, "width": width, "unique": unique}

    @staticmethod
    def _spec_leaves(formula, leaves):
        if type(formula) == AtomicProposition:
            leaves.append(formula.name)
        else:
            for sub in formula._subformula:
                RegexSpec._spec_leaves(sub, leaves)


class Symbol(object):
    def __init__(self, symbol, symbol_type, count = 0):
        self.symbol = {symbol: count}
        self.type = symbol_type
        self.count = count
        self.Xable = True


class SymbolType(Enum):
    Letter = 1
    Option = 2
    Group = 3


class LTLException(Exception):
    def __init__(self):
        pass
