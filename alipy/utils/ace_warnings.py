import warnings


class RepeatElementWarning(UserWarning):
    """
    if contains repeated elements.
    """

class InexistentElementWarning(UserWarning):
    """If attempt to manipulate inexistent element"""


class UnexpectedParameterWarning(UserWarning):
    """
    if unexpected parameter is given.
    """


class ValidityWarning(UserWarning):
    """
    Validity checking fails.
    """


class FunctionWarning(UserWarning):
    """
    Specific function may not be run correctly.
    """
