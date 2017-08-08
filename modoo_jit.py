import os
import sys
import inspect
import re
import hashlib
import functools
import imp
from distutils.extension import Extension
import cython
import Cython
from Cython.Utils import cached_function, get_cython_cache_dir
from Cython.Build.Inline import _get_build_extension, extract_func_code
from Cython.Build.Dependencies import strip_string_literals, cythonize
from Cython.Compiler import Pipeline
from Cython.Compiler.Main import Context, default_options
from Cython.Compiler.TreeFragment import parse_from_strings
from Cython.Compiler.ParseTreeTransforms import CythonTransform, SkipDeclarations, EnvTransform

@cached_function
def _create_context(cython_include_dirs):
    return Context(list(cython_include_dirs), default_options)


def unsafe_type(arg, context=None):
    py_type = type(arg)
    if py_type is int:
        return 'long'
    else:
        return safe_type(arg, context)


def safe_type(arg, context=None):
    py_type = type(arg)
    if py_type in (list, tuple, dict, str):
        return py_type.__name__
    elif py_type is complex:
        return 'double complex'
    elif py_type is float:
        return 'double'
    elif py_type is bool:
        return 'bint'
    elif 'numpy' in sys.modules and isinstance(arg, sys.modules['numpy'].ndarray):
        return 'numpy.ndarray[numpy.%s_t, ndim=%s]' % (arg.dtype.name, arg.ndim)
    else:
        for base_type in py_type.mro():
            if base_type.__module__ in ('__builtin__', 'builtins'):
                return 'object'
            _module = context.find_module(base_type.__module__, need_pxd=False)
            if _module:
                entry = _module.lookup(base_type.__name__)
                if entry.is_type:
                    return '%s.%s' % (base_type.__module__, base_type.__name__)
        return 'object'


_inline_default_context = _create_context(('.',))
_inline_cache = {}
_so_ext = None

def _populate_unbound(args, unbound_sym, local_sym=None, global_sym=None):
    for symbol in unbound_sym:
        if symbol not in args:
            if local_sym is None or global_sym is None:
                calling_frame = inspect.currentframe().f_back.f_back.f_back
                if local_sym is None:
                    local_sym = calling_frame.f_locals
                if global_sym is None:
                    global_sym = calling_frame.f_globals
            if symbol in local_sym:
                args[symbol] = local_sym[symbol]
            elif symbol in global_sym:
                args[symbol] = global_sym[symbol]
            else:
                print("Couldn't find %r" % symbol)


_find_non_space = re.compile('[^ ]').search


def strip_common_indent(code):
    min_indent = None
    lines = code.splitlines()
    for line in lines:
        match = _find_non_space(line)
        if not match:
            continue  # blank
        indent = match.start()
        if line[indent] == '#':
            continue  # comment
        if min_indent is None or min_indent > indent:
            min_indent = indent
    for ix, line in enumerate(lines):
        match = _find_non_space(line)
        if not match or not line or line[indent:indent+1] == '#':
            continue
        lines[ix] = line[min_indent:]
    return '\n'.join(lines)


@cached_function
def unbound_symbols(code, context=None):
    if context is None:
        context = Context([], default_options)
    from Cython.Compiler.ParseTreeTransforms import AnalyseDeclarationsTransform
    tree = parse_from_strings('(tree fragment)', code)
    for phase in Pipeline.create_pipeline(context, 'pyx'):
        if phase is None:
            continue
        tree = phase(tree)
        if isinstance(phase, AnalyseDeclarationsTransform):
            break
    try:
        import builtins
    except ImportError:
        import __builtin__ as builtins
    return tuple(UnboundSymbols()(tree) - set(dir(builtins)))


class UnboundSymbols(EnvTransform, SkipDeclarations):
    def __init__(self):
        CythonTransform.__init__(self, None)
        self.unbound = set()
    def visit_NameNode(self, node):
        if not self.current_env().lookup(node.name):
            self.unbound.add(node.name)
        return node
    def __call__(self, node):
        super(UnboundSymbols, self).__call__(node)
        return self.unbound


def cython_inline(code, get_type=unsafe_type, lib_dir=os.path.join(get_cython_cache_dir(), 'inline'),
                  cython_include_dirs=None, force=False, quiet=False, local_sym=None, global_sym=None, **kwargs):

    if get_type is None:
        get_type = lambda x: 'object'
    ctx = _create_context(tuple(cython_include_dirs)) if cython_include_dirs else _inline_default_context

    # Fast path if this has been called in this session.
    _unbound_symbols = _inline_cache.get(code)
    if _unbound_symbols is not None:
        _populate_unbound(kwargs, _unbound_symbols, local_sym, global_sym)
        args = sorted(kwargs.items())
        arg_sigs = tuple([(get_type(value, ctx), arg) for arg, value in args])
        invoke = _inline_cache.get((code, arg_sigs))
        if invoke is not None:
            arg_list = [arg[1] for arg in args]
            return invoke(*arg_list)

    orig_code = code
    code, literals = strip_string_literals(code)
    if local_sym is None:
        local_sym = inspect.currentframe().f_back.f_back.f_locals
    if global_sym is None:
        global_sym = inspect.currentframe().f_back.f_back.f_globals
    try:
        _inline_cache[orig_code] = _unbound_symbols = unbound_symbols(code)
        _populate_unbound(kwargs, _unbound_symbols, local_sym, global_sym)
    except AssertionError:
        if not quiet:
            # Parsing from strings not fully supported (e.g. cimports).
            print("Could not parse code as a string (to extract unbound symbols).")
    cimports = []
    for name, arg in list(kwargs.items()):
        if arg is cython:
            cimports.append('\ncimport cython as %s' % name)
            del kwargs[name]
    arg_names = sorted(kwargs)
    arg_sigs = tuple([(get_type(kwargs[arg], ctx), arg) for arg in arg_names])
    key = orig_code, arg_sigs, sys.version_info, sys.executable, Cython.__version__
    module_name = "_cython_inline_" + hashlib.md5(str(key).encode('utf-8')).hexdigest()

    if module_name in sys.modules:
        _module = sys.modules[module_name]
    else:
        build_extension = None
        global _so_ext
        if _so_ext is None:
            # Figure out and cache current extension suffix
            build_extension = _get_build_extension()
            _so_ext = build_extension.get_ext_filename('')

        module_path = os.path.join(lib_dir, module_name + _so_ext)

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        if force or not os.path.isfile(module_path):
            cflags = []
            c_include_dirs = []
            qualified = re.compile(r'([.\w]+)[.]')
            for type, _ in arg_sigs:
                m = qualified.match(type)
                if m:
                    cimports.append('\ncimport %s' % m.groups()[0])
                    # one special case
                    if m.groups()[0] == 'numpy':
                        import numpy
                        c_include_dirs.append(numpy.get_include())
                        # cflags.append('-Wno-unused')
            module_body, func_body = extract_func_code(code)
            params = ', '.join(['%s %s' % a for a in arg_sigs])
            module_code = '''
%(module_body)s
%(cimports)s
def __invoke(%(params)s):
%(func_body)s
    return locals()
            ''' % {'cimports': '\n'.join(cimports),
                   'module_body': module_body,
                   'params': params,
                   'func_body': func_body }
            for key, value in literals.items():
                module_code = module_code.replace(key, value)
            pyx_file = os.path.join(lib_dir, module_name + '.pyx')
            fh = open(pyx_file, 'w')
            try:
                fh.write(module_code)
            finally:
                fh.close()
            extension = Extension(
                name = module_name,
                sources = [pyx_file],
                include_dirs = c_include_dirs,
                extra_compile_args = cflags)
            if build_extension is None:
                build_extension = _get_build_extension()
            build_extension.extensions = cythonize([extension], include_path=cython_include_dirs or ['.'], quiet=quiet)
            build_extension.build_temp = os.path.dirname(pyx_file)
            build_extension.build_lib  = lib_dir
            build_extension.run()

        _module = imp.load_dynamic(module_name, module_path)

    _inline_cache[orig_code, arg_sigs] = _module.__invoke
    arg_list = [kwargs[arg] for arg in arg_names]
    return _module.__invoke(*arg_list)


def jit(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        function_code = strip_common_indent(inspect.getsource(func))

        # strip decorators
        new_lines = list()
        for line in function_code.splitlines():
            if len(line) > 0 and '@' in line[0]:
                pass
            else:
                new_lines.append(line)

        function_code = '\n'.join(new_lines) + '\nreturn {}'.format(func.__name__)
        cythonized_function = cython_inline(function_code)
        res = cythonized_function(*args, **kwargs)
        return res

    return wrapper
