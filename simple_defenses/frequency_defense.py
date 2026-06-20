import torch
import torch
import torchaudio
from scipy import signal

# from torch_lfilter import lfilter
# from .lfilter import lfilter


## lfilter --------------------------------------------------------------------


def lfilter(b, a, x):
    """PyTorch lfilter

    Args:
        b (torch.Tensor): The numerator coefficient vector in a 1-D sequence.
        a (torch.Tensor): The denominator coefficient vector in a 1-D sequence.
            if ``a[0]`` is not 1, then both ``a`` and ``b`` are normalized by ``a[0]``.
        x (torch.Tensor): An N-dimensional input tensor.

    Note:
        The filtering happens along dimension (axis) 0.

    """
    y = _LFilter.apply(
        b, a, x.reshape(x.shape[0], -1).to(dtype=torch.float64, device=x.device)
    )
    return y.reshape(*x.shape).to(device=x.device, dtype=x.dtype)


## LFilter --------------------------------------------------------------------


def _lfilter_general_forward(x, y, b, a, order, num_timesteps):
    """ general lfilter implementation valid for all devices """
    y[0] += b[-1] * x[0]
    for n in range(1, order, 1):
        y[n] += (b[-1 - n :] * x[: n + 1]).sum(0)
        y[n] -= (a[-n:] * y[:n]).sum(0)

    for n in range(order, num_timesteps, 1):
        y[n] += (b * x[n - order + 1 : n + 1]).sum(0)
        y[n] -= (a * y[n - order + 1 : n]).sum(0)


def _lfilter_general_backward(dL_dx, dL_dy, b, a, order, num_timesteps):
    """ general lfilter backward implementation valid for all devices """
    for n in range(num_timesteps - 1, order - 1, -1):
        dL_dy[n - order + 1 : n] -= a * dL_dy[n : n + 1]
        dL_dx[n - order + 1 : n + 1] += b * dL_dy[n : n + 1]

    for n in range(order - 1, 0, -1):
        dL_dy[:n] -= a[-n:] * dL_dy[n : n + 1]
        dL_dx[: n + 1] += b[-n - 1 :] * dL_dy[n : n + 1]
    dL_dx[0] += b[-1] * dL_dy[0]


class _LFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, a, x):
        if not (b.ndim == a.ndim == 1):
            raise ValueError("filter vectors b and a should be 1D.")
        b = torch.tensor(
            [float(bb) / float(a[0]) for bb in b], dtype=x.dtype, device=x.device
        )[:, None]
        a = torch.tensor(
            [float(aa) / float(a[0]) for aa in reversed(a[1:])],
            dtype=x.dtype,
            device=x.device,
        )[:, None]
        order = b.shape[0]
        num_timesteps = x.shape[0]
        ctx.save_for_backward(b, a)

        y = torch.zeros_like(x)

        if x.device == torch.device("cpu"):
            _lfilter_forward = _lfilter_general_forward
        elif x.device == torch.device("cuda"):
            if _lfilter_cuda_forward is not None:
                _lfilter_forward = _lfilter_cuda_forward
            else:
                warnings.warn(WARNING_MSG % (x.device, x.device))
                _lfilter_forward = _lfilter_general_forward
        else:
            warnings.warn(WARNING_MSG % (x.device, x.device))
            _lfilter_forward = _lfilter_general_forward

        _lfilter_forward(x, y, b, a, order, num_timesteps)

        return y

    @staticmethod
    def backward(ctx, dL_dy):
        b, a = ctx.saved_tensors
        order = b.shape[0]
        num_timesteps = dL_dy.shape[0]

        dL_dy = dL_dy.clone()  # allow inplace operations on dL_dy
        dL_dx = torch.zeros_like(dL_dy)

        if dL_dy.device == torch.device("cpu"):
            _lfilter_backward = _lfilter_general_backward
        elif dL_dy.device == torch.device("cuda"):
            if _lfilter_cuda_backward is not None:
                _lfilter_backward = _lfilter_cuda_backward
            else:
                _lfilter_backward = _lfilter_general_backward
        else:
            _lfilter_backward = _lfilter_general_backward

        _lfilter_backward(dL_dx, dL_dy, b, a, order, num_timesteps)

        return None, None, dL_dx




class FreqDomainDefense(): 

    def __init__(self, defense_type: str, *args) -> None:

        self.defense_type = defense_type
    
    def __call__(self, x, *args):

        if self.defense_type == 'DS':
            output = DS(x, *args)
        elif self.defense_type == 'LPF':
            output = LPF(x, *args)
        elif self.defense_type == 'BPF':
            output = BPF(x, *args)
        else:
            raise NotImplementedError(f'Unknown defense type: {self.defense_type}!')
        return output

    def _get_name(self, *args):

        if self.defense_type == 'DS':
            name = 'Down_Sampling'
        elif self.defense_type == 'LPF':
            name = 'Low_Pass_Filter'
        elif self.defense_type == 'BPF':
            name = 'Band_Pass_Filter'
        else:
            raise NotImplementedError(f'Unknown defense type: {self.defense_type}!')
        return name

def DS(audio, param=0.5, fs=16000, same_size=True):
    
    assert torch.is_tensor(audio) == True
    assert torch.is_tensor(audio) == True
    ori_shape = audio.shape
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0) # (T, ) --> (1, T)
    elif len(audio.shape) == 2: # (B, T)
        pass
    elif len(audio.shape) == 3:
        audio = audio.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')
    
    down_ratio = param
    new_freq = int(fs * down_ratio)
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=new_freq, resampling_method='sinc_interpolation').to(audio.device)
    up_sampler = torchaudio.transforms.Resample(orig_freq=new_freq, new_freq=fs, resampling_method='sinc_interpolation').to(audio.device)
    down_audio = resampler(audio)
    new_audio = up_sampler(down_audio)
    if same_size: ## sometimes the returned audio may have longer size (usually 1 point)
        return new_audio[..., :audio.shape[1]].view(ori_shape)
    else:
        return new_audio.view(ori_shape[:-1] + new_audio.shape[-1:])

def LPF(new, fs=16000, wp=4000, param=8000, gpass=3, gstop=40, same_size=True, bits=16):

    assert torch.is_tensor(new) == True
    ori_shape = new.shape
    if len(new.shape) == 1:
        new = new.unsqueeze(0) # (T, ) --> (1, T)
    elif len(new.shape) == 2: # (B, T)
        pass
    elif len(new.shape) == 3:
        new = new.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')
    
    if 0.9 * new.max() <= 1 and 0.9 * new.min() >= -1:
        clip_max = 1
        clip_min = -1
    else:
        clip_max = 2 ** (bits - 1) - 1
        clip_min = -2 ** (bits - 1)

    ws = param
    wp = 2 * wp / fs
    ws = 2 * ws / fs
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
    b, a = signal.butter(N, Wn, btype='low', analog=False, output='ba')
    
    audio = new.T.to("cpu") # torch_lfilter only supports CPU tensor speed up
    a = torch.tensor(a, device="cpu", dtype=torch.float) 
    b = torch.tensor(b, device="cpu", dtype=torch.float)
    new_audio = None
    for ppp in range(audio.shape[1]): # torch_lfilter will give weird results for batch samples when using cpu tensor speed up; so we use naive loop here
        new_audio_ = lfilter(b, a, audio[:, ppp:ppp+1]).T
        if new_audio is None:
            new_audio = new_audio_
        else:
            new_audio = torch.cat((new_audio, new_audio_), dim=0)
    new_audio = new_audio.clamp(clip_min, clip_max)
    return new_audio.to(new.device).view(ori_shape)

def BPF(new, fs=16000, wp=[300, 4000], param=[50, 8000], gpass=3, gstop=40, same_size=True, bits=16):

    assert torch.is_tensor(new) == True
    ori_shape = new.shape
    if len(new.shape) == 1:
        new = new.unsqueeze(0) # (T, ) --> (1, T)
    elif len(new.shape) == 2: # (B, T)
        pass
    elif len(new.shape) == 3:
        new = new.squeeze(1) # (B, 1, T) --> (B, T)
    else:
        raise NotImplementedError('Audio Shape Error')

    if 0.9 * new.max() <= 1 and 0.9 * new.min() >= -1:
        clip_max = 1
        clip_min = -1
        # print(clip_max, clip_min)
    else:
        clip_max = 2 ** (bits - 1) - 1
        clip_min = -2 ** (bits - 1)
    
    ws = param
    wp = [2 * wp_ / fs for wp_ in wp]
    ws = [2 * ws_ / fs for ws_ in ws]
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
    b, a = signal.butter(N, Wn, btype="bandpass", analog=False, output='ba', fs=None)

    audio = new.T.to("cpu")
    a = torch.tensor(a, device="cpu", dtype=torch.float)
    b = torch.tensor(b, device="cpu", dtype=torch.float)
    
    new_audio = None
    for ppp in range(audio.shape[1]):
        new_audio_ = lfilter(b, a, audio[:, ppp:ppp+1]).T
        if new_audio is None:
            new_audio = new_audio_
        else:
            new_audio = torch.cat((new_audio, new_audio_), dim=0)
    new_audio = new_audio.clamp(clip_min, clip_max)
    
    return new_audio.to(new.device).view(ori_shape)