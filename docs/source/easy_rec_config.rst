easy_rec config
==============

Dataset parameters 
-----------------------------------------
- ``name (str)`` – Name of the dataset to use, e.g. `ml-1m`, `amazon_beauty`, `behance`. Default to **ml-100k**.
- ``data_folder (str)`` – Path to the raw dataset folder. Default to **../data/raw/**.
- ``min_rating (int or float)`` – Minimum rating threshold. Interactions below this value will be filtered out. Default to **0**.
- ``min_items_per_user (int)`` – Minimum number of items a user must have interacted with. Default to **5**.
- ``min_users_per_item (int)`` – Minimum number of users that must have interacted with an item. Default to **5**.
- ``densify_index (bool)`` – If ``True``, user and item indices will be re-mapped to a contiguous range. Default to **True**.
- ``split_method (str)`` – Method used to split the dataset into train/val/test sets. Options in [`leave_n_out`, `hold_out`, `k_fold`]. Default to **leave_n_out**.
- ``test_sizes (list of int or null)`` – Number of final interactions kept for validation and test sets. Default to **[1, 1]**.
- ``dataset_params``
    - ``split_keys (dict)`` – Keys used to group data for splitting. Default to `{train: [sid, uid], val: [sid, uid], test: [sid, uid]}`.
- ``collator_params``
    - ``sequential_keys (list of str)`` – Keys used to identify sequence order. Default to **[sid]**.
    - ``padding_value (int)`` – Value used to pad sequences. Default to **0**.
    - ``lookback (int)`` – Number of past items to include in each training sample. Default to **200**.
    - ``lookforward (int)`` – Number of future items to predict. Default to **1**.
    - ``simultaneous_lookforward (int)`` – Number of future steps included in a single prediction step. Default to **1**.
    - ``out_seq_len (dict)`` – Output sequence length per split.  
        - ``train``- Default to **null**. 
        - ``val`` - Default to **1** . 
        - ``test``- Default to **1**.
    - ``num_negatives (dict)`` – Number of negative samples per positive example.  
        - ``train`` - Default to  **1**.
        - ``val`` - Default to **100**. 
        - ``test`` - Default to **100**.

    - ``negatives_distribution (str)`` – Strategy for sampling negatives. Default to **uniform**.



Loader parameters
-----------------------------------------
- ``batch_size (int)`` – Number of samples processed in each training batch. Default to **128**.
- ``drop_last (bool)`` – If `True`, discards the last batch if it contains fewer than `batch_size` samples. Default to **True**.
- ``num_workers (int)`` – The number of workers processing the data. Default to **1**.
- ``shuffle (bool)`` – If `True`, shuffles the dataset at every epoch. Default to **True**.
- ``persisent_workers (bool)`` –  If `True`, worker processes remain active between epochs to improve loading speed. Default to **False**.
- ``pin_memory (bool)`` –  If `True`, allocates tensors in pinned memory, which speeds up GPU transfers. Default to **False**.



Training parameters
-----------------------------------------
- ``accelerator (str)`` – Specifies the hardware accelerator to use. Options in [`cpu`, `cuda`, `mps`]. Default set to **cpu**.
- ``enable_checkpointing (bool)`` – Enables or disables automatic checkpoint saving during training. Default to **True**.
- ``max_epochs (int)`` – Maximum number of training epochs. Default to **600**.
- ``log_every_n_steps (int)`` – Number of steps between logging events. Default to **1**.
- ``callbacks``
    - ``ModelCheckpoint``
        - ``dirpath (str)`` – Directory path where checkpoints are saved. Example: `${__exp__.project_folder}out/models/${__exp__.name}/`.
        - ``filename (str)`` – Base name for saved checkpoint files. Default to **best**.
        - ``save_top_k (int)`` – Number of best models to keep. Default to **1**.
        - ``save_last (bool)`` – Whether to save the last checkpoint regardless of performance. Default to **True**.
        - ``monitor (str)`` – Metric to monitor for saving best checkpoints. Example: `val_NDCG_@10/dataloader_idx_0`.
        - ``mode (str)`` – Whether to maximize or minimize the monitored metric. Options in [`min`, `max`]. Default to **max**.
        - ``enable_version_counter (bool)`` – If **False**, overwrites the best model instead of creating versions. Default to **False**.
- ``logger``
    - ``name (str)`` – Logger type to use, e.g. `CSVLogger`, `WandbLogger`. Default to **CSVLogger**.
    - ``save_dir (str)`` – Directory where logs are saved. Example: `${__exp__.project_folder}out/log/${__exp__.name}/`.
    - ``version (int)`` – Logger version. If **0**, overwrites existing logs. Default to **0**.



Model parameters
-----------------------------------------
Common features
~~~~~~~~~~~~~~~~
- ``name (str)`` – Name of the recommender model. Options in [`BERT4Rec`, `Caser`, `CORE`, `CosRec`, `GRU4Rec`, `HGN`, `LightGCN`, `Mamba4Rec`, `NARM`, `NCF`, `SASRec`]
- ``emb_size (int)`` – Size of the items and positions embeddings. Default to **64**.

BERT4Rec
~~~~~~~~~~~~~~~~
- ``bert_num_blocks (int)`` – Number of Transformer blocks in the encoder of BERT4Rec. Default to **2**.
- ``ber_num_heads (int)`` – Number of attention heads in the Transformer model of BERT4Rec. Default to **4**.
- ``dropout_rate (float)`` – Dropout rate for regularization. Default to **0.1**.

Caser
~~~~~~~~~~~~~~~~
- ``lookback (int)`` – Length of the input sequence (number of past time steps considered). Typically sourced from `${data_params.dataset_params.lookback}`.
- ``num_ver_filters (int)`` – Number of vertical convolutional filters. Default to **2**.
- ``num_hor_filters (int)`` – Number of horizontal convolutional filters. Default to **2**.
- ``act_conv (str)`` – Activation function used in the convolutional layers. Default to **Tanh**.
- ``act_fc (str)`` – Activation function used in the fully connected layers. Default to **Tanh**.
- ``drop_rate (float)`` – Dropout rate for regularization. Default to **0.5**.

CORE
~~~~~~~~~~~~~~~~
- ``sess_dropout_rate (float)`` – Dropout rate applied to session representations for regularization. Default to **0.2**.
- ``item_dropout_rate (float)`` – Dropout rate applied to item representations for regularization.  Default to **0.2**.

CosRec
~~~~~~~~~~~~~~~~
- ``block_dims (list of int)`` – Dimensions of convolutional or processing blocks. Default to **[128, 256]**.
- ``fc_dim (int)`` – Dimension of the fully connected layer.  Default to **150**.
- ``act_fc (str)`` – Activation function used in the fully connected layer. Default to **Tanh**.
- ``dropout_rate (float)`` – Dropout rate for regularization. Default to **0.5**.

GRU4Rec
~~~~~~~~~~~~~~~~
- ``num_layers (int)`` – Number of GRU layers. Default to **1**.
- ``dropout_hidden (float)`` – Dropout rate applied to the hidden layers of the GRU. Default to **0.0**.
- ``dropout_input (float)`` – Dropout rate applied to the input embeddings. Default to **0.2**.


HGN
~~~~~~~~~~~~~~~~
- ``lookback (int)`` – Length of the input sequence (number of past time steps considered). Typically sourced from `${data_params.dataset_params.lookback}`.

LightGCN
~~~~~~~~~~~~~~~~
- ``num_layers (int)`` – Number of graph convolution layers applied in LightGCN. Default: **1**.


Mamba4Rec
~~~~~~~~~~~~~~~~
- ``d_model (int)`` – Dimensionality of the model layers and embeddings. Default to **64**.
- ``ssm_cfg.d_model (int)`` – Dimensionality used within the SSM (State Space Model) configuration. Inherits the value from `d_model`.

NARM
~~~~~~~~~~~~~~~~
- ``hidden_size (int)`` – Number of hidden units in the GRU layer. Default to **50**.

- ``n_layers (int)`` – Number of GRU layers used in the attention-based session encoder. Default to **1**.

- ``emb_dropout (float)`` – Dropout rate applied to the input embeddings for regularization. Default to **0.25**.

- ``ct_dropout (float)`` – Dropout rate applied to the context vector or attention mechanism. Default to **0.5**.

NCF
~~~~~~~~~~~~~~~~
- ``mlp_emb_size (int)`` – Embedding size used for the MLP (Multi-Layer Perceptron) component of the model. Default to **8**.

- ``mf_emb_size (int)`` – Embedding size used for the MF (Matrix Factorization) component of the model. Default to **8**.

- ``layers (list of int)`` – List specifying the number of units in each hidden layer of the MLP. Default to **[32, 16, 8]**.

SASRec
~~~~~~~~~~~~~~~~
- ``num_blocks (int)`` – Number of transformer blocks (stacked self-attention + feed-forward layers).  Default to **1**.
- ``num_heads (int)`` – Number of attention heads in the multi-head self-attention layer. Default to **1** .
- ``dropout_rate (float)`` – Dropout rate for regularization. Default to **0.2**.


Global Data Parameters
~~~~~~~~~~~~~~~~
They declare global defaults or shared parameters within the whole config structure.

- ``data_params.collator_params.lookforward (int)`` – Number of future items to look ahead when generating training instances. Default to **0**.
- ``data_params.collator_params.mask_prob (float)`` – Probability of masking each item in the sequence during training. Default to **0.15**
- ``data_params.collator_params.keep_last.train (int)`` – Number of last interactions to keep in each training session. Default to **1**.
- ``data_params.collator_params.keep_last.val (int or null)`` – Number of last interactions to keep for validation. If `null`, no filtering is applied. Default: **null**.
- ``data_params.collator_params.keep_last.test (int or null)`` – Number of last interactions to keep for test.  If `null`, all test interactions are kept. Default: **null**.

Step Routing Parameters 
~~~~~~~~~~~~~~~~
They define how data flows through the model during training, validation and test.

- ``model_input_from_batch (list of str)`` – Specifies which keys from the input batch are passed as inputs to the model. Default to `[in_sid, out_sid]`, where `in_sid` refers to the input sequence IDs, and `out_sid` refers to the target sequence IDs.
- ``loss_input_from_model_output (dict)`` – Defines the inputs to the loss function coming from the model's output or batch.  

   - ``input: null`` indicates that the model output is directly used for loss computation without additional inputs from the batch.