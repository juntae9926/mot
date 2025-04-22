from .bot_sort import BoTSORT

def get_tracker(tracker_name, args):
    tracker_name = tracker_name.lower()
    if tracker_name == "botsort":
        return BoTSORT(args)