import numpy as np

from enum import Enum
from general.singleton import Singleton


class Levenshtein(Singleton):

    def __init__(self):
        self.DELETION_COST = 1
        self.INSERTION_COST = 1
        self.SUBSTITUTION_COST = 1

    def levenshtein_distance(self, obj1, obj2):
        return self.levenshtein_matrix(obj1, obj2)[len(obj1), len(obj2)]

    def _recursive_levenshtein(self, obj1, len1, obj2, len2):
        # base case
        if len1 == 0: return len2
        if len2 == 0: return len1

        # check if last element matches
        sub_cost = 0 if obj1[len1 - 1] == obj2[len2 - 1] else self.SUBSTITUTION_COST

        return min(self.recursive_levenshtein(obj1, len1 - 1, obj2, len2) + self.DELETION_COST,
                   self.recursive_levenshtein(obj1, len1, obj2, len2 - 1) + self.INSERTION_COST,
                   self.recursive_levenshtein(obj1, len1 - 1, obj2, len2 - 1) + sub_cost)

    def levenshtein_matrix(self, obj1, obj2):
        height = len(obj1) + 1
        width = len(obj2) + 1
        matrix = np.ndarray(shape=(height, width))
        for i in range(height):
            matrix[i, 0] = i
        for j in range(1, width):
            matrix[0, j] = j

        for i in range(1, height):
            for j in range(1, width):
                sub_cost = 0 if obj1[i - 1] == obj2[j - 1] else self.SUBSTITUTION_COST
                matrix[i,j] = min(matrix[i-1, j  ] + self.DELETION_COST,    # deletion
                                  matrix[i  , j-1] + self.INSERTION_COST,   # insertion
                                  matrix[i-1, j-1] + sub_cost)              # substitution

        return matrix

    def levenshtein_confusion_matrix(self, objects):
        n_objects = len(objects)
        matrix = np.ndarray(shape=(n_objects, n_objects))
        for i in range(n_objects):
            for j in range(i, n_objects):
                if i == j:
                    matrix[i, j] = 0
                else:
                    matrix[i, j] = self.levenshtein_distance(objects[i], objects[j])
                    matrix[j, i] = matrix[i, j]

        return matrix

    def levenshtein_backtrace(self, matrix):
        # first element is removed, because it is an empty string (due to return string starts with "-")
        return [int(x) for x in self._levenshtein_backtrace(matrix.shape[0] - 1, matrix.shape[1] - 1, matrix).split("-")[1:]]

    def _levenshtein_backtrace(self, row, col, matrix):
        if col > 0 and matrix[row, col - 1] + self.INSERTION_COST == matrix[row, col]:
            return "{0}-{1}".format(self._levenshtein_backtrace(row, col - 1, matrix), EditOperation.Insert.value)

        if row > 0 and matrix[row - 1, col] + self.DELETION_COST == matrix[row, col]:
            return "{0}-{1}".format(self._levenshtein_backtrace(row - 1, col, matrix), EditOperation.Delete.value)

        if row > 0 and col > 0 and matrix[row-1, col-1] + self.SUBSTITUTION_COST == matrix[row, col]:
            return "{0}-{1}".format(self._levenshtein_backtrace(row - 1, col - 1, matrix), EditOperation.Substitute.value)

        if row > 0 and col > 0 and matrix[row-1, col-1] == matrix[row, col]:
            return "{0}-{1}".format(self._levenshtein_backtrace(row - 1, col - 1, matrix), EditOperation.Match.value)

        return ""


class EditOperation(Enum):
    Match = 0
    Delete = 1
    Insert = 2
    Substitute = 3