import inspect
import logging
from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, get_type_hints, get_origin, Literal, get_args, TypedDict, Sequence, Union, TypeAlias, \
    Optional, is_typeddict, Annotated, overload, Iterable

import docstring_parser

logger = logging.getLogger(__name__)


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


ToolParametersProperty: TypeAlias = Union[
    Property, EnumProperty, ArrayProperty, ObjectProperty, DictionaryProperty, dict[str, Any]]

ToolParameters = TypedDict('ToolParameters', {
    'type': Literal['object'],
    'properties': dict[str, ToolParametersProperty],
    'required': Sequence[str],
    '$defs': dict[str, 'ToolParameters'],
}, total=False)


@dataclass
class Tool:
    name: str
    description: str
    parameters: ToolParameters
    func: Callable
    metadata: dict[str, Any]

    @classmethod
    def from_func(cls,
                  func: Callable,
                  *docstring_args,
                  metadata: dict[str, Any] = None,
                  hidden_default_types: Sequence[type] = None,
                  hidden_default_values: Sequence[Any] = None,
                  **docstring_kwargs) -> 'Tool':
        if not metadata:
            metadata = {}

        if hidden_default_types:
            hidden_default_types = tuple(hidden_default_types)

        doc = docstring_parser.parse(inspect.getdoc(func) or '')

        description = doc.short_description or ''
        if doc.long_description:
            description += f'\n\n{doc.long_description}'

        if docstring_args or docstring_kwargs:
            try:
                description = description.format(*docstring_args, **docstring_kwargs)
            except (ValueError, KeyError, IndexError):
                logger.warning(
                    f'Failed to format docstring for tool {func.__name__} with args {docstring_args} and '
                    f'kwargs {docstring_kwargs}. Using unformatted description.',
                    exc_info=True)

        param_docs = {p.arg_name: p.description for p in doc.params}

        sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)

        required: list[str] = []
        properties: dict[str, ToolParametersProperty] = {}

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            is_required = param.default == inspect.Parameter.empty
            if not is_required:
                if hidden_default_values and param.default in hidden_default_values:
                    continue
                if hidden_default_types and isinstance(param.default, hidden_default_types):
                    continue

            python_type = type_hints.get(param_name, Any)
            json_schema = cls._schema_from_type(python_type)

            if param_name in param_docs and param_docs[param_name]:
                param_description = param_docs[param_name]
                if json_schema.get('description'):
                    json_schema['description'] += f'\n\n{param_description}'
                else:
                    json_schema['description'] = param_description

            properties[param_name] = json_schema

            if is_required:
                required.append(param_name)

        raw_parameters = {
            'type': 'object',
            'properties': properties,
            'required': required
        }

        defs = cls._hoist_definitions(raw_parameters)
        if defs:
            raw_parameters['$defs'] = defs

        parameters = ToolParameters(**raw_parameters)

        return cls(name=func.__name__, description=description, parameters=parameters, func=func, metadata=metadata)

    @staticmethod
    def _hoist_definitions(schema: dict[str, Any]) -> dict[str, Any]:
        collected_defs = {}

        def search_and_extract(node: Any):
            if isinstance(node, dict):
                if "$defs" in node:
                    collected_defs.update(node.pop("$defs"))

                for value in node.values():
                    search_and_extract(value)

            elif isinstance(node, list):
                for item in node:
                    search_and_extract(item)

        search_and_extract(schema)
        return collected_defs

    @staticmethod
    def _schema_from_type(t: type) -> ToolParametersProperty:
        origin = get_origin(t)

        # Handle Annotated
        description = None
        if origin is Annotated:
            args = get_args(t)
            t = args[0]
            for arg in args[1:]:
                if isinstance(arg, str):
                    description = arg
                    break

            origin = get_origin(t)

        schema: ToolParametersProperty | None = None

        # Handle Optional (e.g. Optional[str] or Union[str, None])
        if origin is Union or origin is UnionType:
            args = get_args(t)
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                schema = Tool._schema_from_type(non_none_args[0])
            else:
                schema = Tool._schema_from_type(non_none_args[0])
            schema['description'] = description or schema.get('description', '')

        # Pydantic support
        elif hasattr(t, 'model_json_schema') and callable(t.model_json_schema):
            # noinspection PyTypeChecker
            schema = t.model_json_schema()
            schema.pop('title', None)
            schema['description'] = description or schema.get('description', '')

        # Handle basic types
        elif t == str:
            schema = Property(type='string', description=description)
        elif t == int:
            schema = Property(type='integer', description=description)
        elif t == float:
            schema = Property(type='number', description=description)
        elif t == bool:
            schema = Property(type='boolean', description=description)

        # Handle Literals
        elif origin is Literal:
            schema = EnumProperty(type='string', enum=list(get_args(t)), description=description)

        # Handle Lists (e.g. list[str])
        elif t == list or origin == list:
            args = get_args(t)
            item_schema = Tool._schema_from_type(args[0]) if args else {}
            schema = ArrayProperty(type='array', items=item_schema, description=description)

        # Handle TypedDict
        elif is_typeddict(t):
            properties = {}

            # get_type_hints handles inheritance and forward refs better than __annotations__
            hints = get_type_hints(t, include_extras=True)

            for key, value in hints.items():
                properties[key] = Tool._schema_from_type(value)

            required_keys = set(getattr(t, "__required_keys__", []))
            required = []
            for key in required_keys:
                val_type = hints[key]

                while get_origin(val_type) is Annotated:
                    val_type = get_args(val_type)[0]

                origin = get_origin(val_type)
                args = get_args(val_type)

                # Check if the type allows None (Union[..., None] or ... | None)
                is_nullable = (origin is Union or origin is UnionType) and type(None) in args

                # Only keep it required if it is NOT nullable
                if not is_nullable:
                    required.append(key)

            schema = ObjectProperty(
                type='object',
                properties=properties,
                required=required,
                description=description
            )

        # Fallback for dicts or complex types (treat as generic object)
        elif t == dict or origin == dict:
            schema = DictionaryProperty(type='object', additionalProperties=True, description=description)

        if schema is None:
            raise NotImplementedError(f'Type {t} (origin: {origin}) is not supported.')

        return schema


@overload
def tool(func: Callable) -> Tool: ...


@overload
def tool(*docstring_args,
         metadata: dict[str, Any] = None,
         hidden_default_types: Sequence[type] = None,
         hidden_default_values: Sequence[Any] = None,
         **docstring_kwargs) -> Callable[[Callable], Tool]: ...


def tool(func: Callable | Any = None,
         *args,
         metadata: dict[str, Any] = None,
         hidden_default_types: Sequence[type] = None,
         hidden_default_values: Sequence[Any] = None,
         **kwargs) -> Tool | Callable[[Callable], Tool]:
    """
    Decorator to convert a function into a BigTalk Tool.

    Supports both:
      @tool
      def my_func(): ...

      @tool(metadata={'scope': 'read'})
      def my_func(): ...
    """
    is_factory = func is None or not callable(func)

    if is_factory:
        format_args = list(args)
        if func is not None:
            format_args.insert(0, func)

        def wrapper(f: Callable) -> Tool:
            return Tool.from_func(f, metadata=metadata, hidden_default_values=hidden_default_values,
                                  hidden_default_types=hidden_default_types, *format_args, **kwargs)

        return wrapper

    return Tool.from_func(func, metadata=metadata, hidden_default_values=hidden_default_values,
                          hidden_default_types=hidden_default_types, *args, **kwargs)
