# NVFP4 Quantization Guide

Tento dokument popisuje proces kvantizace modelu Qwen3-8B do formátu NVFP4 pro Blackwell GPU (RTX PRO 4000, RTX 5090).

## Obsah

1. [Přehled NVFP4](#přehled-nvfp4)
2. [Pipeline kvantizace](#pipeline-kvantizace)
3. [Parametry kvantizace](#parametry-kvantizace)
4. [Parametry TensorRT engine](#parametry-tensorrt-engine)
5. [SM120 omezení](#sm120-omezení)
6. [Porovnání kvantizačních formátů](#porovnání-kvantizačních-formátů)
7. [Alternativní nástroje](#alternativní-nástroje)

---

## Přehled NVFP4

**NVFP4** (NVIDIA FP4) je 4-bitový floating-point formát exkluzivní pro Blackwell GPU architekturu.

### Výhody oproti INT4

| Vlastnost | NVFP4 | INT4 |
|-----------|-------|------|
| Datový typ | Floating-point | Integer |
| Dynamický rozsah | Zachován (exponent) | Omezený |
| Přesnost | Lepší pro LLM | Nižší |
| HW podpora | Blackwell tensor cores | Starší GPU |
| Komprese | ~4x | ~4x |

### Jak NVFP4 funguje

```
FP16 váha: [mantissa 10bit][exponent 5bit][sign 1bit] = 16 bitů
NVFP4 váha: [mantissa 2bit][exponent 1bit][sign 1bit] = 4 bity

Block-wise quantization (block_size=16):
- 16 vah sdílí společný scale faktor
- Scale je FP16, ukládá se jednou pro celý blok
- Efektivní komprese: (16 × 4 + 16) / (16 × 16) = 31.25% původní velikosti
```

### Konfigurace odpovídající NVIDIA referenci

Náš pipeline používá stejné parametry jako oficiální `nvidia/Qwen3-8B-NVFP4`:

```json
{
  "producer": {"name": "modelopt", "version": "0.35.0"},
  "quantization": {
    "quant_algo": "NVFP4",
    "kv_cache_quant_algo": "FP8",
    "group_size": 16,
    "exclude_modules": ["lm_head"]
  }
}
```

---

## Pipeline kvantizace

```
┌─────────────────────┐
│  HuggingFace Model  │  Qwen/Qwen3-8B (~16GB FP16)
│  (safetensors)      │
└──────────┬──────────┘
           │
           ▼ snapshot_download()
┌─────────────────────┐
│  /models/hf_model   │  Lokální kopie
└──────────┬──────────┘
           │
           ▼ quantize.py (ModelOpt)
┌─────────────────────┐
│  NVFP4 Checkpoint   │  ~4GB kvantizované váhy
│  + FP8 KV cache     │  + kalibrační škály
└──────────┬──────────┘
           │
           ▼ trtllm-build
┌─────────────────────┐
│  TensorRT Engine    │  Optimalizovaný binární engine
│  (.engine)          │  ~6.4GB (včetně runtime bufferů)
└──────────┬──────────┘
           │
           ▼ setup_triton_model_repository()
┌─────────────────────┐
│  Triton Repository  │  Připraveno pro inference
└─────────────────────┘
```

---

## Parametry kvantizace

### `quantize_nvfp4()` - Funkce pro kvantizaci

```python
def quantize_nvfp4(
    checkpoint_dir: Path,      # Vstup: HF model nebo TRT-LLM checkpoint
    output_dir: Path,          # Výstup: kvantizovaný checkpoint
    calib_size: int = 512,     # Počet kalibračních vzorků
    calib_max_seq_length: int = 2048,  # Max délka sekvence při kalibraci
    calib_dataset: str | None = None,  # Custom kalibrační dataset
) -> Path
```

### Podrobný popis parametrů

#### `--qformat nvfp4`

NVIDIA FP4 kvantizační formát.

```
Podporované formáty v TensorRT-LLM 1.2.x:
├── nvfp4      ← Blackwell only (SM120+)
├── fp8        ← Hopper+ (SM90+)
├── int8_sq    ← SmoothQuant (SM80+)
├── int4_awq   ← AWQ (SM80+)
└── full_prec  ← bez kvantizace
```

#### `--kv_cache_dtype fp8`

Kvantizace KV cache do FP8 formátu.

```
Proč FP8 pro KV cache:
- KV cache roste lineárně s délkou sekvence
- Pro 4096 tokenů × 8B model: ~2GB KV cache v FP16
- FP8 redukuje na ~1GB, INT8 na ~1GB
- FP8 má lepší přesnost než INT8 pro attention scores

Varování z ModelOpt:
"Large KV activation detected: 0.55, Quantized KV cache may lead to higher accuracy drop"
→ Některé vrstvy mají vysoké aktivace, FP8 může způsobit mírný pokles přesnosti
```

#### `--calib_size 512`

Počet vzorků pro kalibraci kvantizačních škál.

```
Kalibrace = forward pass přes vzorky → měření rozsahů aktivací

calib_size  | Přesnost | Čas kalibrace
------------|----------|---------------
128         | Nižší    | ~2 min
512         | Dobrá    | ~8 min (default)
1024        | Vyšší    | ~15 min
2048        | Nejlepší | ~30 min

Doporučení: 512 je dobrý kompromis, více není nutné pro NVFP4
```

#### `--calib_max_seq_length 2048`

Maximální délka sekvence během kalibrace.

```
Proč je důležitá:
- Aktivace se liší podle pozice v sekvenci (RoPE)
- Krátká kalibrace (512) nezachytí dlouhé závislosti
- Pro email analýzu (typicky 500-2000 tokenů) je 2048 vhodné

Paměťová náročnost:
calib_max_seq_length × batch_size × model_hidden_size × 2 (gradienty)
2048 × 4 × 4096 × 2 ≈ 64MB na vrstvu
```

#### `--calib_dataset` (volitelný)

Custom dataset pro kalibraci.

```python
# Default: cnn_dailymail (novinové články)
# Pro email analýzu by byl lepší email dataset

Možnosti:
- "cnn_dailymail"     # Default, dobré pro obecný text
- "openwebtext"       # Webový text
- "wikitext"          # Encyklopedický text
- "custom/my-emails"  # Vlastní HuggingFace dataset
```

#### `lm_head` exclusion

Proč není `lm_head` kvantizován:

```
lm_head = poslední lineární vrstva: hidden_state → vocabulary logits

Model: ... → LayerNorm → lm_head (4096 → 152064) → softmax → token

Proč nechat v FP16:
1. Relativně malá (~600MB) vs celkové váhy (~16GB)
2. Přímý vliv na výstupní pravděpodobnosti
3. Kvantizace způsobuje "vocabulary collapse" - model preferuje časté tokeny
4. NVIDIA doporučuje exclude_modules: ["lm_head"]

TensorRT-LLM 1.2.x: lm_head NENÍ kvantizován by default
(Nepotřebujeme --exclude_modules, protože neexistuje)
```

---

## Parametry TensorRT engine

### `build_engine()` - Kompilace do TensorRT

```python
def build_engine(
    checkpoint_dir: Path,
    output_dir: Path,
    max_input_len: int = 4096,
    max_output_len: int = 512,
    max_batch_size: int = 32,
    max_num_tokens: int = 8192,
) -> Path
```

### Detailní popis `trtllm-build` flagů

#### Sekvenční konfigurace

```bash
--max_input_len 4096      # Maximální délka promptu
--max_seq_len 4608        # max_input + max_output
--max_batch_size 32       # Souběžné requesty
--max_num_tokens 8192     # Celkový limit tokenů v batchi
```

```
Memory trade-off:

max_batch_size × max_seq_len × kv_cache_per_token
32 × 4608 × 0.5MB ≈ 73GB → příliš mnoho!

Proto používáme paged KV cache (dynamická alokace)
```

#### NVFP4 optimalizace

```bash
--gemm_plugin nvfp4
```

```
GEMM = General Matrix Multiply (jádro transformeru)

Blackwell FP4 Tensor Cores:
- 2x throughput vs FP8
- 4x throughput vs FP16
- Native HW dekomprese NVFP4 → FP16 pro výpočet
```

#### Attention konfigurace

```bash
--gpt_attention_plugin bfloat16
```

```
Proč BF16 pro attention:
- Softmax vyžaduje vyšší přesnost (exponenciála)
- FP16 může způsobit overflow při velkých sekvencích
- BF16 má stejný rozsah jako FP32

SM120 limitace (ZAKÁZÁNO):
# --context_fmha enable           # FMHA kernel neexistuje
# --use_fp8_context_fmha enable   # FP8 FMHA neexistuje
# --fuse_fp4_quant enable         # Fúze není podporována
```

#### Memory optimalizace

```bash
--kv_cache_type paged     # PagedAttention
--tokens_per_block 64     # Blok = 64 tokenů
--remove_input_padding enable
```

```
PagedAttention:
┌──────────────────────────────────────────┐
│ Tradiční: [seq1][PAD][PAD][seq2][PAD]   │ → plýtvání
│ Paged:    [seq1|seq2|seq3|...]          │ → efektivní
└──────────────────────────────────────────┘

tokens_per_block=64:
- Větší bloky = méně overhead
- Menší bloky = méně fragmentace
- 64 je optimum pro Blackwell L2 cache
```

#### Compute optimalizace

```bash
--use_fused_mlp enable
--norm_quant_fusion enable
```

```
Fused MLP:
Normálně: Linear1 → write → read → SiLU → write → read → Linear2
Fused:    Linear1 + SiLU + Linear2 v jednom kernelu

Norm + Quant fusion:
Normálně: LayerNorm → write → read → Quantize
Fused:    LayerNorm + Quantize v jednom průchodu
```

#### Build optimalizace

```bash
--max_beam_width 1        # Greedy decoding
--workers 1               # Počet kompilačních workerů
--multiple_profiles enable
```

```
multiple_profiles:
- TensorRT vytvoří více optimalizačních profilů
- Runtime vybere nejlepší kernel pro aktuální batch size
- Trade-off: 2x delší build, lepší inference
```

---

## SM120 omezení

### Blackwell architektura

```
SM100 = B100/B200 (datacenter)    → plná NVFP4 podpora
SM120 = RTX 5090/PRO 4000         → omezená FMHA podpora
```

### Co nefunguje na SM120

| Feature | SM100 | SM120 | Důvod |
|---------|-------|-------|-------|
| `--context_fmha` | ✓ | ✗ | FMHA kernel neexistuje |
| `--use_fp8_context_fmha` | ✓ | ✗ | FP8 FMHA kernel neexistuje |
| `--fuse_fp4_quant` | ✓ | ✗ | Fúze není implementována |
| `--gemm_plugin nvfp4` | ✓ | ✓ | Funguje |
| `--norm_quant_fusion` | ✓ | ✓ | Funguje |

### Očekávaná podpora

TensorRT-LLM 1.3+ by měl přidat SM120 FMHA kernely. Sleduj:
- https://github.com/NVIDIA/TensorRT-LLM/releases

---

## Porovnání kvantizačních formátů

### Přehled formátů

| Formát | Bity | GPU | Přesnost | Rychlost |
|--------|------|-----|----------|----------|
| FP16 | 16 | Všechny | Baseline | 1x |
| BF16 | 16 | Ampere+ | ~FP16 | 1x |
| FP8 | 8 | Hopper+ | -0.5% | 1.5x |
| INT8 SQ | 8 | Ampere+ | -1% | 1.3x |
| INT4 AWQ | 4 | Ampere+ | -2% | 1.8x |
| **NVFP4** | 4 | Blackwell | -1% | **2.5x** |

### Memory footprint (Qwen3-8B)

```
FP16:  ~16 GB
FP8:   ~8 GB
INT8:  ~8 GB
INT4:  ~4 GB
NVFP4: ~4 GB (+ lepší přesnost než INT4)
```

---

## Alternativní nástroje

### 1. TensorRT-LLM + ModelOpt (náš přístup)

```bash
# Kvantizace
python quantize.py --qformat nvfp4 --kv_cache_dtype fp8

# Build
trtllm-build --gemm_plugin nvfp4
```

**Výhody:** Nejlepší výkon, NVIDIA supported
**Nevýhody:** Vendor lock-in, binární engine

### 2. LLM Compressor (vLLM)

```python
from llmcompressor import oneshot

oneshot(
    model="Qwen/Qwen3-8B",
    recipe="nvfp4",
    output_dir="./quantized"
)
```

**Výhody:** Safetensors výstup, portabilní
**Nevýhody:** Omezená NVFP4 podpora

### 3. HuggingFace Transformers (v4.51+)

```python
from transformers import AutoModelForCausalLM, NvFp4Config

config = NvFp4Config(group_size=16)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=config
)
```

**Výhody:** Jednoduchá integrace
**Nevýhody:** Nižší výkon než TensorRT

---

## Troubleshooting

### Chyba: `--exclude_modules` not recognized

```
TensorRT-LLM 1.2.x nemá tento parametr.
lm_head NENÍ kvantizován by default.
Řešení: Nepoužívat --exclude_modules
```

### Varování: `Large KV activation detected`

```
ModelOpt varuje, že některé aktivace jsou vysoké.
FP8 KV cache může způsobit mírný pokles přesnosti.
Řešení: Akceptovatelné, nebo použít INT8 KV cache
```

### Chyba: `FMHA kernel not found for SM120`

```
SM120 nemá FMHA kernely pro NVFP4+FP8.
Řešení: Zakázat --context_fmha, --use_fp8_context_fmha
```

### Chyba: `PermissionError: model.cache`

```
trtllm-build potřebuje zapisovat timing cache.
Řešení: chown -R appuser:appuser /app
```

---

## Reference

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [trtllm-build flags](https://nvidia.github.io/TensorRT-LLM/latest/commands/trtllm-build.html)
- [NVFP4 Technical Blog](https://developer.nvidia.com/blog/accelerating-inference-with-nvfp4-on-blackwell/)
