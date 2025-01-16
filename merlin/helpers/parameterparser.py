from lark import Lark, Transformer


class ParameterParser:
    """
    ParameterParser is a class that parses a string of parameters into a nested dictionary structure.
    
    Methods:
        parse(param_str: str) -> dict: Parses the given parameter string and returns a nested dictionary.
    Usage:
        parser = ParameterParser()
        result = parser.parse("param1=value1,param2.subparam=value2")
        # result will be {'param1': 'value1', 'param2': {'subparam': 'value2'}}
    """
    grammar = """
        start: attr ("," attr)*
        attr: attr_name "=" attr_value
        attr_name: CNAME ("." CNAME)*
        attr_value: quoted_string | unquoted_string | FLOAT | INT

        quoted_string: ESCAPED_STRING
        unquoted_string: /[a-zA-Z0-9_]+/ 

        // Terminals
        %import common.CNAME
        %import common.ESCAPED_STRING
        %import common.SIGNED_FLOAT -> FLOAT
        %import common.SIGNED_INT -> INT
        %import common.WS
        %ignore WS
    """
    class AttrExtractionTransformer(Transformer):
        def quoted_string(self, value):
            return value[0][1:-1]  # Remove quotes

        def FLOAT(self, value):
            return float(value[0])

        def INT(self, value):
            return int(value[0])

        def unquoted_string(self, value):
            return value[0].value

        def attr_name(self, value):
            return value

        def attr(self, children):
            attr_name, attr_value = children
            return (attr_name, attr_value)

        def start(self, children):
            result = {}
            for attr_name, attr_value in children:
                attr_name = [
                    part.value
                    for part in attr_name
                ]
                current = result
                for key in attr_name[:-1]: 
                    current = current.setdefault(key, {})
                current[attr_name[-1]] = attr_value
            return result
        
        def attr_value(self, value):
            return value[0]

    def __init__(self):
        self._parser = Lark(
            self.grammar, 
            parser="lalr", 
            transformer=self.AttrExtractionTransformer())

    def parse(self, param_str):
        return self._parser.parse(param_str)