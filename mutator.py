import ast
import copy
from typing import Iterable, List, Any

class Marker(ast.NodeTransformer):
    """Mark the target statement."""
    def __init__(self, line_no: int) -> None:
        super().__init__()

        self.line_no = line_no
        self.found = False          # target found?
        self.loop_level = 0         # depth of loop (0 indicates outside loop body)
        self.is_first_stmt = False  # is the first stmt in block?

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.stmt) and node.lineno == self.line_no:
            setattr(node, '__target__', (self.loop_level > 0, self.is_first_stmt))
            self.found = True
            return node
        
        if isinstance(node, ast.stmt): #add condition
            self.is_first_stmt = True
        if isinstance(node, (ast.While, ast.For, ast.AsyncFor)): #get in to loop
            self.loop_level += 1
        for field, old_value in ast.iter_fields(node): #no change
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        if isinstance(node, ast.stmt): #get out
            self.is_first_stmt = False
        if isinstance(node, (ast.While, ast.For, ast.AsyncFor)): #get out of loop
            self.loop_level -= 1
        return node


def mk_abstract() -> ast.expr:
    """Create an AST for the abstract condition."""
    return ast.Name('__abstract__')

class MutationOperator(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.mutated = False

class Tighten(MutationOperator):
    """If the target statement is an if-statement, transform its condition by
    conjoining an abstract condition: if c => if c and not __abstract__."""
    def __init__(self) -> None:
        super().__init__()
    
    def visit(self, node: ast.AST) -> Any:
        try:
            if isinstance(node, ast.If) and node.__target__: #node is if and there is need for change
                abstract = ast.BoolOp(op=ast.And())
                added = ast.UnaryOp(op=ast.Not(), operand=mk_abstract())
                abstract.values = [node.test, added]
                node.test = abstract
                self.mutated = True
        except:
            pass
        return super().visit(node)
    

class Loosen(MutationOperator):
    """If the target statement is an if-statement, transform its condition by
    disjoining an abstract condition: if c => if c or __abstract__."""
    def __init__(self) -> None:
        super().__init__()
    
    def visit(self, node: ast.AST) -> Any:
        try:
            if isinstance(node, ast.If) and node.__target__: #node is if and there is need for change
                abstract = ast.BoolOp(op=ast.Or())
                added = mk_abstract()
                abstract.values = [node.test, added]
                node.test = abstract
                self.mutated = True
        except:
            pass
        return super().visit(node)


class Guard(MutationOperator):
    """Transform the target statement so that it executes only if an abstract condition is false:
    s => if not __abstract__: s."""
    def visit(self, node: ast.AST) -> Any:
        try:
            if node.__target__: #There is need for change
                abstract = ast.If()
                abstract.test = ast.UnaryOp(op=ast.Not(), operand=mk_abstract())
                abstract.body = node
                abstract.orelse = []
                self.mutated = True
                node = abstract
                return node
        except AttributeError:
            pass
        return super().visit(node)


class Break(MutationOperator):
    """If the target statement is in loop body, right before it insert a `break` statement that
    executes only if an abstract condition is true, i.e., if __abstract__: break."""
    def __init__(self, required_position: bool) -> None:
        """If `required_position` is `True`, this operation is performed only when the 
        target is the first statement.
        If `required_position` is `False`, this operation is performed only when the 
        target is not the first statement.
        """
        super().__init__()
        self.required_position = required_position
        self.is_avoid = False #so far only case while-else

    def visit(self, node: ast.AST) -> Any:
        try:
            if node.__target__[1] == self.required_position and node.__target__[0] and not self.is_avoid: #only under loop
                abstract = ast.If()
                abstract.test = mk_abstract()
                abstract.body = ast.Break()
                abstract.orelse = []
                self.mutated = True
                added_string = ast.unparse(abstract) + "\n" + ast.unparse(node)
                node = ast.parse(added_string)
                return node   
        except AttributeError:
            pass
        return super().visit(node)

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if field == "orelse": #technically outside loop
                self.is_avoid = True
            else:
                self.is_avoid = False
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node


class Mutator:
    """Perform program mutation."""
    def __init__(self, tree: ast.Module, line_no: int, log: bool = False) -> None:
        assert isinstance(tree, ast.Module)
        self.old_tree = tree
        self.log = log
        
        marker = Marker(line_no)
        self.marked_tree = marker.visit(copy.deepcopy(tree))
        assert marker.found

    def apply(self, ops: List[MutationOperator] = None) -> Iterable[ast.Module]:
        if ops is None:
            # in default priority order
            ops = [Tighten(), Loosen(), Break(True), Guard(), Break(False)] 

        for visitor in ops:
            new_tree = visitor.visit(copy.deepcopy(self.marked_tree))
            if self.log:
                print(f'-> {visitor.__class__.__name__}', '✓' if visitor.mutated else '✗')

            if visitor.mutated:
                if self.log:
                    print(ast.unparse(new_tree)) 
                yield new_tree
