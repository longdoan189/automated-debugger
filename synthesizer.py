import ast
import copy
import itertools
from types import CodeType
from typing import Any, Dict, List, Tuple
from tester import *

class Template:
    """Condition template."""
    def __init__(self, meta_vars: List[Tuple[str, str]], body: ast.expr) -> None:
        self.args: List[str] = [s for s, _ in meta_vars]
        self.types: List[str] = [t for _, t in meta_vars]
        self.body = body

    @classmethod
    def from_lambda(cls, expr: str):
        """Parse a template from a string, basic syntax:
        (<var> : <type>, <var> : <type>, ...) => <condition>
        The lhs parenthesis is optional."""
        lhs, body = expr.split('=>')
        lhs = lhs.strip()
        if lhs.startswith('(') and lhs.endswith(')'):
            lhs = lhs[1:-1]
        lhs = lhs.split(',') if ',' in lhs else [lhs]

        meta_vars = []
        for x in lhs:
            arg, typ = x.split(':')
            meta_vars.append( (arg.strip(), typ.strip()) )
        
        body = ast.parse(body.strip()).body[0].value
        assert isinstance(body, ast.expr)
        return cls(meta_vars, body)

    def instantiate(self, vars: List[str]) -> ast.expr:
        """Instantiate a template with `vars`. Just like applying values to lambda expressions."""
        assert len(vars) == len(self.args)
        return self.NameTransformer(dict(zip(self.args, vars))).visit(copy.deepcopy(self.body))

    class NameTransformer(ast.NodeTransformer):
        def __init__(self, mapping: Dict[str, str]) -> None:
            super().__init__()
            self.mapping = mapping

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id in self.mapping:
                node.id = self.mapping[node.id]
                return node

            return node

class Synthesizer:
    """Condition synthesis."""

    def __init__(self, tree: ast.Module, pos_tests: List[CodeType], neg_tests: List[CodeType],
                 k: int = 10, extra_templates: List[Template] = None, log: bool = False) -> None:
        self.abstract_tree = tree
        self.pos_tests = pos_tests
        self.neg_tests = neg_tests
        self.k = k
        self.extra_templates = extra_templates
        self.log = print if log else no_log

        self.condition: ast.expr            # synthesized condition
        self.concrete_tree: ast.Module      # instantiated tree
    
    def apply(self) -> bool:
        """Entry point."""
        overall_record = Record([], [])

        for test_code in self.neg_tests:
            self.log(f'--> Execute {test_code.co_name}')

            success, record = exec_abstract(self.abstract_tree, test_code, iter([]))
            self.log(f'--(0/{self.k})->', '✓' if success else '✗', record.values)

            i = 1
            while not success and i <= self.k:
                if i == self.k: # the last attempt
                    future_values = all_true()
                else:
                    future_values = iter(self.flip(record.values))

                success, record = exec_abstract(self.abstract_tree, test_code, future_values)
                self.log(f'--({i}/{self.k})->', '✓' if success else '✗', record.values)
                i += 1

            if success:
                overall_record += record
            else:
                return False # synthesis failed

        for test_code in self.pos_tests:
            self.log(f'--> Execute {test_code.co_name} (+)')

            success, record = exec_abstract(self.abstract_tree, test_code, iter([]))
            self.log('   ', '✓' if success else '✗', record.values)
            
            assert success
            overall_record += record
        
        if self.solve(overall_record):
            self.concrete_tree = Instantiate(self.condition).visit(
                copy.deepcopy(self.abstract_tree))
            return True
        else:
            return False

    def flip(self, values: List[bool]) -> List[bool]:
        """Flip the last `False` to `True`, and drop all the `True`s after the last `False`."""
        if len(values) == 0:
            return []
        if values[-1]:
            return self.flip(values[:-1])
        else:
            return values[:-1] + [True]

    def replace_variable(self, node:ast.AST, replace_map: dict) ->ast.AST:
        """Replace value from grammar template to current value"""
        if isinstance(node, (ast.Constant, ast.FormattedValue,ast.JoinedStr, ast.List, ast.Tuple,ast.Set,ast.Dict,ast.Pass,ast.alias, ast.Break, ast.Continue, ast.MatchStar)):
            return node
        elif isinstance(node, (ast.Module, ast.Interactive, ast.Expression, ast.FunctionType)):
            for e in range(len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
        elif isinstance(node, (ast.Name)):
            node.id = replace_map[node.id]
        elif isinstance(node, (ast.Expr,ast.Starred, ast.Attribute)):
            node.value = self.replace_variable(node.value, replace_map)
        elif isinstance(node, ast.UnaryOp):
            node.operand = self.replace_variable(node.operand, replace_map)
        elif isinstance(node, ast.BinOp):
            node.left = self.replace_variable(node.left, replace_map)
            node.right = self.replace_variable(node.right, replace_map)
        elif isinstance(node, ast.BoolOp):
            for e in range (len(node.values)):
                node.values[e] = self.replace_variable(node.values[e], replace_map)
        elif isinstance(node, ast.Compare):
            node.left = self.replace_variable(node.left, replace_map)
            for e in range (len(node.comparators)):
                node.comparators[e] = self.replace_variable(node.comparators[e], replace_map)
        elif isinstance(node, ast.Call):
            for e in range (len(node.args)):
                node.args[e] = self.replace_variable(node.args[e], replace_map)
        elif isinstance(node, ast.keyword):
            node.arg = self.replace_variable(node.arg, replace_map)
            node.value = self.replace_variable(node.value, replace_map)
        elif isinstance(node, ast.IfExp):
            node.test = self.replace_variable(node.test, replace_map)
            node.body = self.replace_variable(node.body, replace_map)
            node.orelse = self.replace_variable(node.orelse, replace_map)
        elif isinstance(node, ast.NamedExpr):
            node.target = self.replace_variable(node.target, replace_map)
            node.value = self.replace_variable(node.value, replace_map)
        elif isinstance(node, ast.Subscript):
            node.value = self.replace_variable(node.value, replace_map)
            node.slice = self.replace_variable(node.slice, replace_map)
        elif isinstance(node, ast.Slice):
            node.lower = self.replace_variable(node.lower, replace_map)
            node.upper = self.replace_variable(node.upper, replace_map)
            node.step = self.replace_variable(node.step, replace_map)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            node.elt = self.replace_variable(node.elt, replace_map)
            for e in range (len(node.generators)):
                node.generators[e] = self.replace_variable(node.generators[e], replace_map)
        elif isinstance(node, (ast.DictComp)):
            node.key = self.replace_variable(node.key, replace_map)
            node.value = self.replace_variable(node.value, replace_map)
            for e in range (len(node.generators)):
                node.generators[e] = self.replace_variable(node.generators[e], replace_map)
        elif isinstance(node, (ast.comprehension)):
            node.target = self.replace_variable(node.target, replace_map)
            node.iter = self.replace_variable(node.iter, replace_map)
            for e in range (len(node.ifs)):
                node.ifs[e] = self.replace_variable(node.ifs[e], replace_map)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for e in range (len(node.targets)):
                node.targets[e] = self.replace_variable(node.targets[e], replace_map)
            node.value = self.replace_variable(node.value, replace_map)
        elif isinstance(node, ast.Raise):
            node.exc = self.replace_variable(node.exc, replace_map)
            node.cause = self.replace_variable(node.cause, replace_map)
        elif isinstance(node, ast.Assert):
            node.test = self.replace_variable(node.test, replace_map)
            node.msg = self.replace_variable(node.msg, replace_map)
        elif isinstance(node, ast.Delete):
            for e in range(len(node.targets)):
                node.targets[e]  = self.replace_variable(node.targets[e], replace_map)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for e in range(len(node.names)):
                node.names[e]  = self.replace_variable(node.names[e], replace_map)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            node.target = self.replace_variable(node.target, replace_map)
            node.iter = self.replace_variable(node.iter, replace_map)
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
            for e in range (len(node.orelse)):
                node.orelse[e] = self.replace_variable(node.orelse[e], replace_map)
        elif isinstance(node, (ast.If,ast.While)):
            node.test = self.replace_variable(node.test, replace_map)
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
            for e in range (len(node.orelse)):
                node.orelse[e] = self.replace_variable(node.orelse[e], replace_map)
        elif isinstance(node, (ast.Try)):
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
            for e in range (len(node.handlers)):
                node.handlers[e] = self.replace_variable(node.handlers[e], replace_map)
            for e in range (len(node.finalbody)):
                node.finalbody[e] = self.replace_variable(node.finalbody[e], replace_map)
            for e in range (len(node.orelse)):
                node.orelse[e] = self.replace_variable(node.orelse[e], replace_map)
        elif isinstance(node, (ast.ExceptHandler)):
            node.type = self.replace_variable(node.type, replace_map)
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for e in range (len(node.items)):
                node.items[e] = self.replace_variable(node.items[e], replace_map)
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
        elif isinstance(node, (ast.withitem)):
            node.context_expr = self.replace_variable(node.context_expr, replace_map)
            node.optional_vars = self.replace_variable(node.optional_vars, replace_map)
        elif isinstance(node, (ast.Match)):
            node.subject = self.replace_variable(node.subject, replace_map)
            for e in range (len(node.cases)):
                node.cases[e] = self.replace_variable(node.cases[e], replace_map)
        elif isinstance(node, (ast.match_case)):
            node.pattern = self.replace_variable(node.pattern, replace_map)
            node.guard = self.replace_variable(node.guard, replace_map)
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
        elif isinstance(node, (ast.MatchValue, ast.MatchSingleton)):
            node.value = self.replace_variable(node.value, replace_map)
        elif isinstance(node, (ast.MatchSequence, ast.MatchOr)):
            for e in range (len(node.patterns)):
                node.patterns[e] = self.replace_variable(node.patterns[e], replace_map)
        elif isinstance(node, (ast.MatchMapping)):
            for e in range (len(node.keys)):
                node.keys[e] = self.replace_variable(node.keys[e], replace_map)
            for e in range (len(node.patterns)):
                node.patterns[e] = self.replace_variable(node.patterns[e], replace_map)
        elif isinstance(node, (ast.MatchClass)):
            node.cls = self.replace_variable(node.cls, replace_map)
            for e in range (len(node.patterns)):
                node.patterns[e] = self.replace_variable(node.patterns[e], replace_map)
        elif isinstance(node, (ast.MatchAs)):
            node.pattern = self.replace_variable(node.pattern, replace_map)
            node.name = self.replace_variable(node.name, replace_map)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.args = self.replace_variable(node.args, replace_map)
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
            for e in range (len(node.returns)):
                node.returns[e] = self.replace_variable(node.returns[e], replace_map)
        elif isinstance(node, (ast.Lambda)):
            node.args = self.replace_variable(node.args, replace_map)
            node.body = self.replace_variable(node.body, replace_map)
        elif isinstance(node, (ast.arguments)):
            for e in range (len(node.posonlyargs)):
                node.posonlyargs[e] = self.replace_variable(node.posonlyargs[e], replace_map)
            for e in range (len(node.args)):
                node.args[e] = self.replace_variable(node.args[e], replace_map)
            for e in range (len(node.kwonlyargs)):
                node.kwonlyargs[e] = self.replace_variable(node.kwonlyargs[e], replace_map)
            node.vararg = self.replace_variable(node.vararg, replace_map)
            node.kwarg = self.replace_variable(node.kwarg, replace_map)
        elif isinstance(node, (ast.arg)):
            node.annotation = self.replace_variable(node.annotation, replace_map)
        elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
            node.value = self.replace_variable(node.value, replace_map)
        elif isinstance(node, (ast.ClassDef)):
            for e in range (len(node.bases)):
                node.bases[e] = self.replace_variable(node.bases[e], replace_map)
            for e in range (len(node.keywords)):
                node.keywords[e] = self.replace_variable(node.keywords[e], replace_map)
            for e in range (len(node.body)):
                node.body[e] = self.replace_variable(node.body[e], replace_map)
            for e in range (len(node.decorator_list)):
                node.decorator_list[e] = self.replace_variable(node.decorator_list[e], replace_map)
        elif isinstance(node, (ast.Await)):
            node.value = self.replace_variable(node.value, replace_map)
        return node
    
    def mutate_intertype(self,list_mutator):
        '''Helper function to make single dictionary'''
        if len(list_mutator) == 1:
            return list_mutator[0]
        list_1 = list_mutator[-1]
        list_2 = list_mutator[-2]
        combined_list = []
        for each_dict1 in list_1:
            for each_dict2 in list_2:
                combined_list.append(each_dict1 | each_dict2)
        remain_list = list_mutator[:-2]
        remain_list.append(combined_list)
        return self.mutate_intertype(remain_list)
    def mutate_by_type(self, from_input, to_input):
        '''Replace value from grammar with current value'''
        mutator_list = []
        for e in from_input:
            mutator_by_type = []
            list_from = list(from_input[e])
            try:
                list_to = list(to_input[e])
            except KeyError:
                continue
            permutation_to = list(itertools.permutations(list_to))
            for e_p in permutation_to:
                mutator_dict = dict()
                for e in range(len(list_from)):
                    mutator_dict[list_from[e]] = e_p[e]
                mutator_by_type.append(mutator_dict)
            mutator_list.append(mutator_by_type)
        return self.mutate_intertype(mutator_list)
                

    def solve(self, constraints: Record) -> bool:
        """Solve constraints. Returns if a solution is found. Set `self.condition` when found."""
        for each_pair in constraints.envs:  #simple
            for each in each_pair:
                left = ast.Name(id=each)
                right = ast.Constant(each_pair[each])
                expr = ast.Compare(left=left, ops=[ast.Eq()], comparators=[right])
                if self.sat(expr, constraints):
                    self.condition = expr
                    return True
                expr = ast.Compare(left=left, ops=[ast.NotEq()], comparators=[right])
                if self.sat(expr, constraints):
                    self.condition = expr
                    return True
        if self.extra_templates is None:
            return False
        for each in self.extra_templates:
            list_arg, list_type = each.args, each.types
            dict_arg_syntax = dict()
            for e in range(len(list_arg)):
                try:
                    dict_arg_syntax[list_type[e]].add(list_arg[e])
                except KeyError:
                    dict_arg_syntax[list_type[e]] = {list_arg[e]}
            dict_act_original = dict()
            for e in constraints.envs:
                for val in e:
                    type_val = type(e[val]).__name__
                    try:
                        dict_act_original[type_val].add(val)
                    except KeyError:
                        dict_act_original[type_val] = {val}
            
            list_dict_replace = self.mutate_by_type(dict_arg_syntax, dict_act_original)

            for dict_replace in list_dict_replace:
                experiment_body = copy.deepcopy(each.body)
                expr = self.replace_variable(experiment_body,dict_replace)
                if self.sat(expr, constraints):
                    self.condition = expr
                    return True
        return False
        
    def sat(self, cond: ast.expr, constraints: Record) -> bool:
        """Check if `cond` satisfies the `constraints`."""
        for env, value in zip(constraints.envs, constraints.values):
            try:
                actual = eval(ast.unparse(cond), {}, env)
            except Exception:
                return False

            if actual != value:
                return False
        return True
    
    def validate(self) -> bool:
        """Check correctness of the repaired program `self.concrete_tree`."""
        return run_tests(self.concrete_tree, self.neg_tests + self.pos_tests)

def no_log(self, *values: object) -> None:
    pass

