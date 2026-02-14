import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints, get_origin, Literal, get_args, TypedDict, Sequence, Union, TypeAlias, \
    Optional, is_typeddict, Annotated

import docstring_parser


class Property(TypedDict):
    type: Literal['string', 'integer', 'boolean', 'number', 'array', 'object']
    description: Optional[str]


class EnumProperty(Property):
    type: Literal['string']
    enum: Sequence[str]


class ArrayProperty(Property):
    type: Literal['array']
    items: dict[str, Any]


class ObjectProperty(Property):
    type: Literal['object']
    properties: dict[str, Any]
    required: Sequence[str]


class DictionaryProperty(Property):
    type: Literal['object']
    additionalProperties: bool


ToolParametersProperty: TypeAlias = Union[Property, EnumProperty, ArrayProperty, ObjectProperty, DictionaryProperty]


class ToolParameters(TypedDict):
    type: Literal['object']
    properties: dict[str, ToolParametersProperty]
    required: Sequence[str]


@dataclass
class Tool:
    name: str
    description: str
    parameters: ToolParameters
    func: Callable

    @classmethod
    def from_func(cls, func: Callable) -> 'Tool':
        doc = docstring_parser.parse(inspect.getdoc(func) or '')

        description = doc.short_description or ''
        if doc.long_description:
            description += f'\n\n{doc.long_description}'

        param_docs = {p.arg_name: p.description for p in doc.params}

        sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)

        required: list[str] = []
        properties: dict[str, ToolParametersProperty] = {}

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            # Map Python Type -> JSON Schema
            python_type = type_hints.get(param_name, Any)
            json_schema = cls._schema_from_type(python_type)

            # Inject Description from Docstring
            if param_name in param_docs and param_docs[param_name]:
                param_description = param_docs[param_name]
                if json_schema.get('description'):
                    json_schema['description'] += f'\n\n{param_description}'
                else:
                    json_schema['description'] = param_description

            # Add to properties
            properties[param_name] = json_schema

            # Check if required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        parameters = ToolParameters(
            type='object',
            properties=properties,
            required=required
        )

        return cls(name=func.__name__, description=description, parameters=parameters, func=func)

    @staticmethod
    def _schema_from_type(t: type) -> ToolParametersProperty:
        # Handle Annotated
        description = None
        if get_origin(t) is Annotated:
            args = get_args(t)
            t = args[0]
            for arg in args[1:]:
                if isinstance(arg, str):
                    description = arg
                    break

        # Handle basic types
        if t == str:
            return Property(type='string', description=description)
        elif t == int:
            return Property(type='integer', description=description)
        elif t == float:
            return Property(type='number', description=description)
        elif t == bool:
            return Property(type='boolean', description=description)

        # Handle Literals
        if get_origin(t) is Literal:
            return EnumProperty(type='string', enum=list(get_args(t)), description=description)

        # Handle Lists (e.g. list[str])
        if t == list or get_origin(t) == list:
            args = get_args(t)
            item_schema = Tool._schema_from_type(args[0]) if args else {}
            return ArrayProperty(type='array', items=item_schema, description=description)

        # Handle TypedDict
        if is_typeddict(t):
            properties = {}
            required = []

            # get_type_hints handles inheritance and forward refs better than __annotations__
            hints = get_type_hints(t, include_extras=True)

            for key, value in hints.items():
                properties[key] = Tool._schema_from_type(value)

                # TypedDicts usually mark all keys required unless using total=False
                # We assume required for tool use safety, unless Optional is detected
                if get_origin(value) is not Union or type(None) not in get_args(value):
                    required.append(key)

            return ObjectProperty(
                type='object',
                properties=properties,
                required=required,
                description=description
            )

        # Fallback for dicts or complex types (treat as generic object)
        if t == dict or get_origin(t) == dict:
            return DictionaryProperty(type='object', additionalProperties=True, description=description)

        raise NotImplementedError(f'Type {t} is not supported in Tool parameter type mapping.')


def tool(func: Callable) -> Tool:
    """Decorator to convert a function into a BigTalk Tool."""
    return Tool.from_func(func)
