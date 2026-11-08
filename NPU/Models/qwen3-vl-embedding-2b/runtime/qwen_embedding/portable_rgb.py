"""Portable integer equivalent of FFmpeg's unscaled SSSE3 YUV420 RGB path.

Arithmetic references (FFmpeg n8.1, LGPL-2.1-or-later):
https://github.com/FFmpeg/FFmpeg/blob/n8.1/libswscale/x86/yuv_2_rgb.asm
https://github.com/FFmpeg/FFmpeg/blob/n8.1/libswscale/yuv2rgb.c
This implementation uses NumPy integer operations, not copied assembly.
"""
import numpy as np

_COEFF={0:(104597,132201,25675,53279),1:(117489,138438,13975,34925),2:(104597,132201,25675,53279),3:(104597,132201,25675,53279),4:(104448,132798,24759,53109),5:(104597,132201,25675,53279),6:(104597,132201,25675,53279),7:(117579,136230,16907,35559),9:(110013,140363,12277,42626)}


def _coefficients(colorspace,full_range):
    coefficients=list(_COEFF[colorspace]);coefficients[2]=-coefficients[2];coefficients[3]=-coefficients[3]
    if full_range:
        coefficients=[(abs(c)*224//255)*(1 if c>=0 else -1) for c in coefficients];cy=65536;offset=0
    else:cy=65536*255//219;offset=16*8
    quantize=lambda n:(n*8192+32768)>>16
    cr,cb,cg_u,cg_v=map(quantize,coefficients)
    return quantize(cy),offset,cr,cb,cg_u,cg_v


_CPP=None


def convert_yuv420(yuv,height,width,colorspace=1,full_range=False,implementation='numpy'):
    if height<=0 or width<=0 or height%2 or width%2 or colorspace not in _COEFF:raise ValueError('Unsupported portable RGB conversion layout/colorspace')
    shape=yuv.shape[:-2];flat=np.asarray(yuv,dtype=np.uint8).reshape(*shape,-1);size=height*width
    cy,offset,cr,cb,cg_u,cg_v=_coefficients(colorspace,full_range)
    if implementation=='cpp':
        import ctypes
        from pathlib import Path
        global _CPP
        if _CPP is None:
            _CPP=ctypes.CDLL(str(Path(__file__).with_name('portable_rgb_core.so'))).rgb420_compat
            _CPP.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t]+[ctypes.c_int]*8;_CPP.restype=ctypes.c_int
        raw=np.ascontiguousarray(yuv,dtype=np.uint8);assert raw.shape[-2:]==(height*3//2,width)
        output=np.empty((*shape,height,width,3),dtype=np.uint8)
        code=_CPP(raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),raw.size//(size*3//2),height,width,cy,offset,cr,cb,cg_u,cg_v)
        if code:raise ValueError('Invalid fast RGB layout')
        return output
    if implementation!='numpy':raise ValueError('Unknown RGB implementation')
    y=flat[...,:size].reshape(*shape,height,width).astype(np.int32)
    u=flat[...,size:size+size//4].reshape(*shape,height//2,width//2).astype(np.int32)
    v=flat[...,size+size//4:size+size//2].reshape(*shape,height//2,width//2).astype(np.int32)
    luma=((y*8-offset)*cy)>>16;u=(u-128)*8;v=(v-128)*8
    expand=lambda x:np.repeat(np.repeat(x,2,axis=-2),2,axis=-1)
    r=luma+expand((v*cr)>>16);b=luma+expand((u*cb)>>16)
    g=luma+expand(((u*cg_u)>>16)+((v*cg_v)>>16))
    return np.stack([r,g,b],axis=-1).clip(0,255).astype(np.uint8)
