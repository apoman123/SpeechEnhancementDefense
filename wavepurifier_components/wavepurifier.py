from .configs import parse_args_and_config
from .diffmodel import RevGuidedDiffusion

def get_wavepurifier(args, config):
    # args, config = parse_args_and_config()
    runner_diff = RevGuidedDiffusion(args, config, device=config.device)
    return runner_diff