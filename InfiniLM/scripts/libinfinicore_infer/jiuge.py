from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref, cast


class JiugeMetaCStruct(Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_uint),
    ]


class JiugeWeightsCStruct(Structure):
    _fields_ = [
        ("nlayer", c_size_t),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("transpose_linear_weights", c_int),
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)),
        ("attn_qkv", POINTER(c_void_p)),
        ("attn_qkv_b", POINTER(c_void_p)),
        ("attn_q_norm", POINTER(c_void_p)),
        ("attn_k_norm", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]


class JiugeModelCStruct(Structure):
    pass


@register_model
class JiugeModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        lib.createJiugeModel.restype = POINTER(JiugeModelCStruct)
        lib.createJiugeModel.argtypes = [
            POINTER(JiugeMetaCStruct),
            POINTER(JiugeWeightsCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.destroyJiugeModel.argtypes = [POINTER(JiugeModelCStruct)]

        lib.createKVCache.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            DataType,
            DeviceType,
            POINTER(c_int),
            c_size_t,
        ]
        lib.createKVCache.restype = POINTER(KVCacheCStruct)

        lib.dropKVCache.argtypes = [POINTER(KVCacheCStruct)]

        lib.inferBatchJiuge.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.forwardBatchJiuge.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            c_void_p,
        ]

        lib.setJiugeFusedFFN.argtypes = [POINTER(JiugeModelCStruct), c_int]
        lib.getJiugeFFNProfile.argtypes = [
            POINTER(JiugeModelCStruct),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_int),
        ]

    def create_model(self, meta, weights, device_type, ndev, dev_ids):
        return self.lib.createJiugeModel(meta, weights, device_type, ndev, dev_ids)

    def destroy_model(self, model):
        self.lib.destroyJiugeModel(model)

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        self.lib.dropKVCache(kv_cache)

    def infer_batch(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_caches,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchJiuge(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_caches,
            temperature,
            topk,
            topp,
            output,
        )

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
    ):
        self.lib.forwardBatchJiuge(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
        )

    def set_fused_ffn(self, model, use_fused):
        self.lib.setJiugeFusedFFN(model, 1 if use_fused else 0)

    def get_ffn_profile(self, model):
        total_ms = c_float(0)
        n_layers = c_int(0)
        # First call to get n_layers (pass NULL for per_layer_ms)
        self.lib.getJiugeFFNProfile(model, byref(total_ms), POINTER(c_float)(), byref(n_layers))
        per_layer = (c_float * max(n_layers.value, 1))()
        self.lib.getJiugeFFNProfile(model, byref(total_ms), per_layer, byref(n_layers))
        return {
            "total_ms": total_ms.value,
            "per_layer_ms": [per_layer[i] for i in range(n_layers.value)],
            "n_layers": n_layers.value,
        }
