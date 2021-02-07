is_simple_core = False

if is_simple_core:
    from deep3framework.core_simple import Variable
    from deep3framework.core_simple import Function
    from deep3framework.core_simple import using_config
    from deep3framework.core_simple import no_grad
    from deep3framework.core_simple import as_array
    from deep3framework.core_simple import as_variable
    from deep3framework.core_simple import setup_variable
else:
    from deep3framework.core import Variable
    from deep3framework.core import Function
    from deep3framework.core import using_config
    from deep3framework.core import no_grad
    from deep3framework.core import as_array
    from deep3framework.core import as_variable
    from deep3framework.core import setup_variable

setup_variable()