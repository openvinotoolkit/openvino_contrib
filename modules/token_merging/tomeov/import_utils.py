def is_diffusers_available():
    try:
        import diffusers
        return True
    except ImportError:
        print("diffusers library is not available. Please install it to use Token Merging.")
        return False

def is_openclip_available():
    try:
        import open_clip
        return True
    except ImportError:
        print("OpenCLIP library is not available. Please install it to use Token Merging.")
        return False

def is_timm_available():
    try:
        import timm
        return True
    except ImportError:
        print("Timm library is not available. Please install it to use Token Merging.")
        return False
