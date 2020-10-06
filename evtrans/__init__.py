"""Network Grafting Algorithm."""


def log_history(log_path, variables, idx, header=None):
    """Logging the history over loss and metrics.
    Everything in CSV split by ",".
    The header is optional
    # Arguments
    log_path: str
        the absolute path of the logging data
    variables: list
        list contains the variables
    idx: int
        usually epoch number or iteration number
    header: list
        header description for the variables
    """
    # Wrtie the first epoch
    if idx == 1:
        if header is not None:
            assert type(header) is list

            with open(log_path, "w") as f:
                header_text = ",".join(header)
                f.write(header_text)
                f.write("\n")
                f.close()
        else:
            # erase potential contents from previous session
            with open(log_path, "w") as f:
                f.close()

    str_vars = [str(var) for var in variables]
    with open(log_path, "a+") as f:
        # prepare logging variables to string
        curr_history = str(idx)+","+",".join(str_vars)
        f.write(curr_history)
        f.write("\n")
        f.close()
