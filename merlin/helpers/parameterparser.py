from lark import Lark, Transformer, v_args


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

    grammar = r"""
        start: attr ("," attr)*
        attr: attr_name "=" attr_value
        attr_name: CNAME ("." CNAME)*
        attr_value: bool | quoted_string | unquoted_string | FLOAT | INT

        quoted_string: ESCAPED_STRING
        unquoted_string: /[a-zA-Z0-9_\/\.\-]+/
        bool: "true" | "false"

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

        def bool(self, value):
            return value[0] == "true"

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
                attr_name = [part.value for part in attr_name]
                current = result
                for key in attr_name[:-1]:
                    current = current.setdefault(key, {})
                current[attr_name[-1]] = attr_value
            return result

        def attr_value(self, value):
            return value[0]

    def __init__(self):
        self._parser = Lark(
            self.grammar, parser="lalr", transformer=self.AttrExtractionTransformer()
        )

    def parse(self, param_str):
        param_str = param_str.strip()
        if param_str == "":
            return {}
        return self._parser.parse(param_str)


class FunctionParser:
    grammar = r"""
        start: function_call

        // A top-level "function_call" like: name(arg1, arg2=..., arg3)
        function_call: unquoted_string "(" [args] ")"

        // Zero or more comma-separated arguments
        args: argument ("," argument)*

        // Try unnamed_argument first so "param" won't be misread as "param="
        ?argument: unnamed_argument | named_argument

        named_argument: attr_name "=" attr_value
        unnamed_argument: attr_value

        attr_name: unquoted_string ("." unquoted_string)*
        attr_value: unquoted_string | quoted_string | FLOAT | INT

        quoted_string: ESCAPED_STRING
        unquoted_string: /[a-zA-Z0-9_\/\.\-]+/

        %import common.ESCAPED_STRING
        %import common.SIGNED_FLOAT -> FLOAT
        %import common.SIGNED_INT -> INT
        %import common.WS
        %ignore WS
    """

    class AttrExtractionTransformer(Transformer):
        # Make "args" produce a list instead of a Tree
        def args(self, children):
            # children is a list of arguments, each argument is from named_argument or unnamed_argument
            return children

        def quoted_string(self, value):
            string = value[0].value[1:-1]  # remove surrounding quotes

            if string in ["true", "false"]:
                return string == "true"

            return string

        def unquoted_string(self, value):
            string = value[0].value

            if string in ["true", "false"]:
                return string == "true"

            return string

        def FLOAT(self, value):
            return float(value[0])

        def INT(self, value):
            return int(value[0])

        # Named argument returns (key, val)
        @v_args(inline=True)
        def named_argument(self, key, val):
            return (key, val)

        # Unnamed argument returns (“_args”, val)
        @v_args(inline=True)
        def unnamed_argument(self, val):
            return ("args", val)

        def start(self, children):
            # Unwrap outer Tree, return just the function_call result
            return children[0]

        def attr_value(self, children):
            # Unwrap inner value from Tree
            return children[0]

        def attr_name(self, children):
            # Unwrap name from Tree
            return children[0]

        def function_call(self, children):
            name = children[0]
            arg_list = (
                children[1] if len(children) > 1 and children[1] is not None else []
            )

            params = {"args": []}
            for key, val in arg_list:
                if key == "args":
                    params["args"].append(val)
                else:
                    params[key] = val
            return (name, params)

    def parse(self, param_str: str):
        parser = Lark(
            self.grammar, parser="lalr", transformer=self.AttrExtractionTransformer()
        )
        return parser.parse(param_str)
