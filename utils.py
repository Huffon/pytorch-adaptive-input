def epoch_time(start_time, end_time):
    """
    Calculate the time spent during one epoch
    
    Args:
        start_time (float): training start time
        end_time   (float): training end time
    
    Returns:
        (int, int) elapsed_mins and elapsed_sec spent during one epoch
    """

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs