def run_parser(func):
    def wrapper(self, *args, **kwargs):
        cmd_args = vars(self.parser.parse_args())
        return func(self, *args, **kwargs, **cmd_args) or 0

    return wrapper


def add_arguments(init, arguments):
    def wrapper(self, parser, *args, **kwargs):
        arguments(self, parser)
        return init(self, parser, *args, **kwargs)

    return wrapper


class ParserRunnerType(type):
    def __new__(mcs, name, bases, attr):
        if "handle" in attr:
            attr["handle"] = run_parser(attr["handle"])

        if "add_arguments" in attr:
            attr["__init__"] = add_arguments(attr["__init__"], attr["add_arguments"])

        return super(ParserRunnerType, mcs).__new__(mcs, name, bases, attr)


class BaseCommand(object, metaclass=ParserRunnerType):
    def __init__(self, parser):
        self.parser = parser
        super().__init__()

    def handle(self, *args, **kwargs):
        print("kwargs", args, kwargs)
        raise NotImplementedError("Command class must implement handle function")
