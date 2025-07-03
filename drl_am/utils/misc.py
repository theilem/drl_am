import distutils.util
import json
import os
import os.path
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError


@dataclass
class DecayParams:
    decay_type: str = "constant"  # exp, linear, cosine, constant
    base: float = 1e-3
    decay_rate: float = 0.1
    decay_steps: int = 1_000_000

def get_decay_curve(decay_params, optimizer_to_step_ratio=1.0):
    import tensorflow as tf
    if decay_params.decay_type == "exp":
        return tf.keras.optimizers.schedules.ExponentialDecay(
            decay_params.base, int(decay_params.decay_steps * optimizer_to_step_ratio), decay_params.decay_rate
        )
    elif decay_params.decay_type == "linear":
        return tf.keras.optimizers.schedules.PolynomialDecay(
            decay_params.base, int(decay_params.decay_steps * optimizer_to_step_ratio),
            end_learning_rate=decay_params.decay_rate * decay_params.base
            , power=1.0
        )
    elif decay_params.decay_type == "cosine":
        return tf.keras.experimental.CosineDecay(
            decay_params.base, int(decay_params.decay_steps * optimizer_to_step_ratio),
            alpha=decay_params.decay_rate
        )  ## TODO: Check if this is the correct implementation
    elif decay_params.decay_type == "constant":
        return decay_params.base
    raise ValueError(f"Unknown decay type: {decay_params.decay_type}")


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')

    try:
        return getattr_recursive_(obj, split)
    except KeyError:
        split.insert(0, 'params')
        return getattr_recursive_(obj, split)


def getattr_recursive_(obj, split):
    if isinstance(obj, dict):
        if len(split) > 1:
            return getattr_recursive(obj[split[0]], split[1:])
        else:
            return obj[split[0]]
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if not isinstance(s, list):
        s = s.split('/')

    if isinstance(obj, dict):
        if not s[0] in obj:
            s.insert(0, 'params')
        if len(s) > 1:
            return setattr_recursive(obj[s[0]], s[1:], val)
        else:
            obj[s[0]] = val
            return None
    if not hasattr(obj, s[0]):
        s.insert(0, 'params')
    return setattr_recursive(getattr(obj, s[0]), s[1:], val) if len(s) > 1 else setattr(obj, s[0], val)


def get_bool_user(message, default: bool):
    resp = input(f'{message} {"[Y/n]" if default else "[y/N]"}\n')
    try:
        return distutils.util.strtobool(resp)
    except ValueError:
        return default


def override_params(params, overrides):
    for override in overrides:
        try:
            oldval = getattr_recursive(params, override[0])
            if type(oldval) == bool:
                to_val = bool(distutils.util.strtobool(override[1]))
            else:
                to_val = type(oldval)(override[1])
            setattr_recursive(params, override[0],
                              to_val)
            print("Overriding param", override[0], "from", oldval, "to", to_val)
        except (KeyError, AttributeError):
            print("Could not override", override[0], "as it does not exist. Aborting.")
            exit(1)

    return params


def load_config(param_class, config_path, overrides=None):
    with open(config_path, 'r') as f:
        js = json.load(f)
    try:
        params = param_class(**js)
    except ValidationError as e:
        print("Some errors occurred while parsing the config file:")
        for error in e.errors():
            print(json.dumps(error, indent=4))
        print("Aborting.")
        exit(1)

    if overrides is not None:
        params = override_params(params, overrides)

    return params


def save_params(params: BaseModel, config_path, force=False):
    if not force and os.path.exists(config_path):
        resp = get_bool_user(f"File exists at {config_path}. Override?", False)
        if not resp:
            print("Chose not to override.")
            return
    js = params.model_dump()

    with open(config_path, 'w') as f:
        json.dump(js, f, indent=4)

    print(f"Saved params to {config_path}.")


def dict_to_tensor(d):
    import tensorflow as tf
    return {key: dict_to_tensor(value) if isinstance(value, dict) else tf.convert_to_tensor(value) for key, value in
            d.items()}


def dict_mean(dict_list):
    mean_dict = {}
    elem1 = dict_list[0]
    for key in elem1.keys():
        if isinstance(elem1[key], str):
            continue
        try:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        except KeyError:
            continue
    return mean_dict


def create_log_dir(config_path, run_id):
    if run_id is None:
        run_id = config_path.split("/")[-1].split(".json")[0]

    log_dir = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}/"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = log_dir + 'models/'
    os.makedirs(model_dir, exist_ok=True)
    return log_dir


def setup_gpu(kwargs):
    import tensorflow as tf
    if not kwargs['gpu'] and kwargs['gpu_id'] is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            gpu_id = int(kwargs['gpu_id']) if kwargs['gpu_id'] is not None else 0
            gpu_used = physical_devices[gpu_id]
            tf.config.set_visible_devices(gpu_used, 'GPU')
            tf.config.experimental.set_memory_growth(gpu_used, True)
            print('Using following GPU: ', gpu_used.name)
        except Exception as e:
            import traceback
            print("Invalid device or cannot modify virtual devices once initialized. Not too good probably")
            print(traceback.print_exception(e))
            exit(0)
            pass


def find_config_model(model, exit_on_not_found=True):
    files = [f"{model}/config.json", f"logs/{model}/config.json", f"example/{model}/config.json"]
    return find_file(model, files, exit_on_not_found)


def find_file(name, files, exit_on_not_found=True):
    for file in files:
        if Path(file).is_file():
            return file
    print(f"Could not find {name}, tried all of {files}")
    if exit_on_not_found:
        exit(1)
    return None


def dict_slice_set(d, idx, assign):
    for key, value in d.items():
        value[idx] = assign[key]
    return d
