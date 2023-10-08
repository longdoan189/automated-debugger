import ast
import inspect
from utils import get_positive_tests, get_negative_tests
from repairer import Repairer
from synthesizer import Template
import inputfunc, inputfunc_tests

def repair(input_module, line_no: int, test_module, extra_template=[]):
    tree = ast.parse(inspect.getsource(input_module))

    pos_tests = get_positive_tests(test_module)
    neg_tests = get_negative_tests(test_module)

    r = Repairer(tree, line_no, pos_tests, neg_tests, log=True, extra_templates=extra_template)
    r.repair()
    r.validate(failures=[])

repair(inputfunc, 9, inputfunc_tests, extra_template=[Template.from_lambda('(a: str, b: str, c: str) => a == b or a == c')])