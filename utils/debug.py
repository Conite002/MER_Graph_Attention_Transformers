
def debug_message(message, variable=None):
    if variable is not None:
        print(f"[DEBUG] {message}: {variable}")
    else:
        print(f"[DEBUG] {message}")