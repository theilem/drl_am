import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention

def create_model(input_space, output_space, output_activation,
                 layer_size=256,
                 num_attention_layers=3,
                 num_layers=3,
                 num_heads=16,
                 key_dim=16,
                 mix_state_into_constraints=True,
                 mix_state_into_goals=True,
                 model_name=None,
                 num_layers_pre_attention=1,
                 num_self_attention_layers=0,
                 layer_norm_before_skip=False
                 ):
    """
    Creates a Keras model based on the provided input and output spaces.
    Args:
        input_space (dict): Dictionary defining the input space with keys like "internal_state", "constraint", etc.
        output_space (int or dict): The output space, can be an integer or a dictionary of output spaces.
        output_activation (str): Activation function for the output layer.
        layer_size (int): Size of the hidden layers.
        num_attention_layers (int): Number of attention layers to use.
        num_layers (int): Total number of dense layers in the model.
        num_heads (int): Number of heads in the multi-head attention layer.
        key_dim (int): Dimension of the key in the attention mechanism.
        mix_state_into_constraints (bool): Whether to mix state into constraints.
        mix_state_into_goals (bool): Whether to mix state into goals.
        model_name (str, optional): Name of the model.
        num_layers_pre_attention (int): Number of dense layers before attention.
        num_self_attention_layers (int): Number of self-attention layers to apply.
        layer_norm_before_skip (bool): Whether to apply layer normalization before skip connections.
    Returns:
        Model: A Keras model instance.
    """
    inputs = {"internal_state": Input(shape=input_space["internal_state"], dtype=tf.float32,
                                      name="internal_state_in")}  # [B, S]

    state_layer = inputs["internal_state"]

    if "constraint" in input_space:
        inputs["constraint"] = Input(shape=input_space["constraint"], dtype=tf.float32,
                                     name="constraint_in")  # [B, C]
        state_layer = tf.concat((state_layer, inputs["constraint"]), axis=-1)

    if "goal" in input_space:
        inputs["goal"] = Input(shape=input_space["goal"], dtype=tf.float32, name="goal_in")  # [B, G]
        state_layer = tf.concat((state_layer, inputs["goal"]), axis=-1)

    state_layer = Dense(layer_size, activation='relu')(state_layer)  # [B, layer_size]

    mul_constraints_layer = None
    mul_constraints_mask = None
    if "mult_constraints" in input_space:
        inputs["mult_constraints"] = Input(shape=input_space["mult_constraints"], dtype=tf.float32,
                                           name="mult_constraints_in")  # [B, c, Cm]
        inputs["num_constraints"] = Input(shape=(1,), dtype=tf.int32, name="num_constraints_in")
        mul_constraints_mask = tf.sequence_mask(inputs["num_constraints"],
                                                maxlen=tf.shape(inputs["mult_constraints"])[1],
                                                dtype=tf.bool)

        mul_constraints_layer = Dense(layer_size, activation='relu')(inputs["mult_constraints"])
        mul_constraints_layer = mul_constraints_layer + state_layer[:, None, :]  # [B, c, layer_size]
        for _ in range(num_layers_pre_attention - 1):
            mul_constraints_layer = Dense(layer_size, activation='relu')(mul_constraints_layer)

        for _ in range(num_self_attention_layers):
            mul_constraints_layer = LayerNormalization()(mul_constraints_layer)
            residual = mul_constraints_layer
            mul_constraints_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, attention_axes=1)(
                query=mul_constraints_layer, value=mul_constraints_layer, attention_mask=mul_constraints_mask)
            mul_constraints_layer = mul_constraints_layer + residual
            mul_constraints_layer = tf.keras.activations.relu(mul_constraints_layer)

        mul_constraints_layer = Dense(layer_size, activation='tanh')(mul_constraints_layer)
        mul_constraints_layer = mul_constraints_layer[:, None, :, :]  # [B, 1, c, layer_size]
        mul_constraints_mask = mul_constraints_mask[:, None, None, :, :]

    mul_goals_layer = None
    mul_goals_mask = None
    if "mult_goals" in input_space:
        inputs["mult_goals"] = Input(shape=input_space["mult_goals"], dtype=tf.float32,
                                     name="mult_goals_in")  # [B, g, Gm]
        inputs["num_goals"] = Input(shape=(1,), dtype=tf.int32, name="num_goals_in")
        mul_goals_mask = tf.sequence_mask(inputs["num_goals"], maxlen=tf.shape(inputs["mult_goals"])[1],
                                          dtype=tf.bool)

        mul_goals_layer = Dense(layer_size, activation='relu')(inputs["mult_goals"])
        mul_goals_layer = mul_goals_layer + state_layer[:, None, :]  # [B, g, layer_size]
        for _ in range(num_layers_pre_attention - 1):
            mul_goals_layer = Dense(layer_size, activation='relu')(mul_goals_layer)

        for _ in range(num_self_attention_layers):
            mul_goals_layer = LayerNormalization()(mul_goals_layer)
            residual = mul_goals_layer
            mul_goals_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, attention_axes=1)(
                query=mul_goals_layer, value=mul_goals_layer, attention_mask=mul_goals_mask)
            mul_goals_layer = mul_goals_layer + residual
            mul_goals_layer = tf.keras.activations.relu(mul_goals_layer)

        mul_goals_layer = Dense(layer_size, activation='tanh')(mul_goals_layer)
        mul_goals_layer = mul_goals_layer[:, None, :, :]  # [B, 1, g, layer_size]
        mul_goals_mask = mul_goals_mask[:, None, None, :, :]

    assert not ("latent" in input_space and "action" in input_space), "Cannot have both latent and action inputs."
    state_layer = state_layer[:, None, :]  # [B, 1, layer_size]
    latent_layer = None
    if "latent" in input_space:
        inputs["latent"] = Input(shape=input_space["latent"], dtype=tf.float32, name="latent_in")  # [B, n, A]
        latent_layer = Dense(layer_size, activation='relu')(inputs["latent"])

        state_layer += latent_layer  # [B, n, layer_size]

    action_layer = None
    if "action" in input_space:
        inputs["action"] = Input(shape=input_space["action"], dtype=tf.float32, name="action_in")  # [B, n, A]
        action_layer = Dense(layer_size, activation='relu')(inputs["action"])

        state_layer += action_layer  # [B, n, layer_size]

    state_layer = LayerNormalization()(state_layer)
    state_layer = state_layer[:, :, None, :]  # [B, [n/1], 1, layer_size]

    for _ in range(num_attention_layers):
        if mul_constraints_layer is not None:
            attn_axes = (1, 2)  # trigger broadcasting if constraint layer indep. of action/latents
            if mix_state_into_constraints:
                mul_constraints_layer += state_layer
                attn_axes = 2  # Avoid cross attention between actions/latents
            mul_constraints_layer = Dense(layer_size, activation='relu')(
                mul_constraints_layer)  # [B, [n/1], c, layer_size]
            mul_constraints_layer = LayerNormalization()(mul_constraints_layer)

            residual = state_layer
            state_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, attention_axes=attn_axes)(
                query=state_layer, value=mul_constraints_layer, attention_mask=mul_constraints_mask)
            if layer_norm_before_skip:
                state_layer = LayerNormalization()(state_layer)
            state_layer = state_layer + residual
            state_layer = tf.keras.activations.relu(state_layer)
            state_layer = LayerNormalization()(state_layer)

        if mul_goals_layer is not None:
            attn_axes = (1, 2)  # trigger broadcasting if goal layer indep. of action/latents
            if mix_state_into_goals:
                mul_goals_layer += state_layer
                attn_axes = 2  # Avoid cross attention between actions/latents
            mul_goals_layer = Dense(layer_size, activation='relu')(mul_goals_layer)  # [B, [n/1], g, layer_size]
            mul_goals_layer = LayerNormalization()(mul_goals_layer)

            residual = state_layer
            state_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, attention_axes=attn_axes)(
                query=state_layer, value=mul_goals_layer, attention_mask=mul_goals_mask)
            if layer_norm_before_skip:
                state_layer = LayerNormalization()(state_layer)
            state_layer = state_layer + residual
            state_layer = tf.keras.activations.relu(state_layer)
            state_layer = LayerNormalization()(state_layer)

    state_layer = tf.squeeze(state_layer, axis=2)  # [B, [n/1], layer_size]
    if action_layer is None and latent_layer is None:
        state_layer = tf.squeeze(state_layer, axis=1)  # [B, layer_size]
    for _ in range(num_layers - 1):
        state_layer = Dense(layer_size, activation='relu')(state_layer)

    if isinstance(output_space, int):
        outputs = Dense(output_space, activation=output_activation)(state_layer)
    elif isinstance(output_space, dict):
        outputs = {}
        for key, space in output_space.items():
            outputs[key] = Dense(space, activation=output_activation)(state_layer)
    else:
        raise ValueError("Invalid output_space. Must be int or dict.")

    return Model(inputs=inputs, outputs=outputs, name=model_name)
