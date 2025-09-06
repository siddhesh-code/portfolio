def retry_on_exception(retries=3, delay=1):
    """Decorator for retrying tests on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
