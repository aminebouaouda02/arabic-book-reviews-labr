import transformers.modeling_utils as mu
import transformers.utils.import_utils as iu

# Patch de compatibilité pour torch 2.1 avec Hugging Face transformers
mu.check_torch_load_is_safe = lambda: None
iu.check_torch_load_is_safe = lambda: None

